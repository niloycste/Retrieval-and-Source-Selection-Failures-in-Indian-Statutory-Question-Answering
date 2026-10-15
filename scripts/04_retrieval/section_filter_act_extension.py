"""Fixed-protocol, train-only Act prediction and section-pool controls.

Run from either repository root. CPU only; never overwrites an existing run.
python scripts/04_retrieval/section_filter_act_extension.py
python scripts/04_retrieval/section_filter_act_extension.py --verify PATH
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline

from citation_specified_questions import residual_tokens
from retrieval_diagnostic_controls import (
    BM25, SECTION_RE, load_jsonl, normalize, parse_section, rank,
    retrieval_metrics, tokenize,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'results/experiments/retrieval/runs/section_filter_act_extension_v2'


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(rows, a, b):
    """b-a, question-weighted mean after resampling whole source clusters."""
    groups = defaultdict(list)
    for r in rows:
        groups[r['group_id']].append(r['r5'][b] - r['r5'][a])
    sums = np.array([sum(v) for v in groups.values()])
    sizes = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(42)
    idx = rng.integers(len(sums), size=(10000, len(sums)))
    draws = 100 * sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    return {'difference_points': float(100*sums.sum()/sizes.sum()),
            'source_cluster_95_interval': np.quantile(draws, [.025, .975]).tolist(),
            'clusters': len(groups), 'iterations': 10000, 'seed': 42}


def verify(directory):
    summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    for split in ('val', 'test'):
        rows = load_jsonl(directory/f'{split}_per_query.jsonl')
        assert len({r['id'] for r in rows}) == len(rows)
        for name, scores in summary[split]['retrieval'].items():
            for k in (1, 3, 5, 10):
                expected = round(100*sum(r['gold_index'] in r['rankings'][name][:k]
                                         for r in rows)/len(rows), 4)
                assert abs(scores[f'recall@{k}']-expected) < 1e-8, (split,name,k)
            rr = []
            for r in rows:
                head = r['rankings'][name][:10]
                rr.append(1/(head.index(r['gold_index'])+1) if r['gold_index'] in head else 0)
                assert len(head) == len(set(head))
                assert all(0 <= i < summary['corpus_n'] for i in head)
                assert r['r5'][name] == int(r['gold_index'] in head[:5])
            assert abs(scores['mrr@10']-round(100*np.mean(rr),4)) < 1e-8
        for name in ('section_random_expected', 'size_only_random_expected'):
            for k in (1, 3, 5, 10):
                probs = []
                for r in rows:
                    pool = r['section_pool']
                    if r['section_fallback']:
                        p = int(r['gold_index'] in r['rankings']['paragraph_bm25'][:k])
                    elif name == 'section_random_expected':
                        p = int(r['gold_index'] in pool)*min(1,k/len(pool))
                    else:
                        p = min(k,len(pool))/summary['corpus_n']
                    probs.append(p)
                assert abs(summary[split][name][f'recall@{k}']-100*np.mean(probs)) < 1e-8
        assert abs(summary[split]['act_classifier']['accuracy_pct']-
                   100*np.mean([r['predicted_act']==r['gold_act'] for r in rows])) < 1e-8
    print('Independent saved-ranking checks passed for validation and test.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
        return
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = {s: ROOT/f'data/splits/{s}.jsonl' for s in ('train','val','test')}
    paths['corpus'] = ROOT/'data/processed/indic_ipr_qa/unique_contexts.jsonl'
    paths['protocol'] = ROOT/'docs/SECTION_FILTER_ACT_EXTENSION_PROTOCOL.md'
    paths['script'] = Path(__file__).resolve()
    paths['bm25_helper'] = Path(__file__).with_name('retrieval_diagnostic_controls.py')
    paths['group_helper'] = Path(__file__).with_name('citation_specified_questions.py')
    save(out/'manifest.json', {'status':'started', 'utc':datetime.now(timezone.utc).isoformat(),
         'inputs':{k:{'path':str(p.relative_to(ROOT)), 'sha256':sha(p)} for k,p in paths.items()},
         'python':platform.python_version(), 'numpy':np.__version__,
         'sklearn':sklearn.__version__, 'joblib':joblib.__version__,
         'protocol':paths['protocol'].read_text(encoding='utf-8')})
    data = {s:load_jsonl(paths[s]) for s in ('train','val','test')}
    audit = {}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        overlap = set(r['group_id'] for r in data[a]) & set(r['group_id'] for r in data[b])
        context_overlap = set(normalize(r['context']) for r in data[a]) & set(normalize(r['context']) for r in data[b])
        assert not overlap and not context_overlap, (a,b,'split leakage')
        qa = set(normalize(r['question']).lower() for r in data[a])
        audit[f'{a}_{b}'] = {'section_overlap':0, 'context_overlap':0,
            'exact_question_overlap_n':sum(normalize(r['question']).lower() in qa for r in data[b])}
    save(out/'split_audit.json', audit)
    corpus = load_jsonl(paths['corpus'])
    paragraphs = [normalize(r['context']) for r in corpus]
    contexts = {}
    for i,p in enumerate(paragraphs):
        contexts.setdefault(p,i)
    sections = [parse_section(r['title']) for r in corpus]
    all_sections = sorted(set(sections)-{None})
    by_section = {s:[i for i,v in enumerate(sections) if v==s] for s in all_sections}
    acts = [r['act'] for r in corpus]
    bm25 = BM25([tokenize(p) for p in paragraphs])
    title_bm25 = BM25([tokenize(r['title']+' '+p) for r,p in zip(corpus,paragraphs)])
    # The source split is section-disjoint but generated question strings can
    # still repeat. Remove every exact held-out query from the classifier's
    # training inputs. No held-out labels or evidence fields are read here.
    heldout_questions = {
        normalize(r['question']).lower()
        for split in ('val', 'test') for r in data[split]
    }
    classifier_train = [r for r in data['train']
                        if normalize(r['question']).lower() not in heldout_questions]
    assert classifier_train and len(classifier_train) < len(data['train'])
    model = make_pipeline(TfidfVectorizer(ngram_range=(1,2), lowercase=True,
        token_pattern=r'(?u)\b\w\w+\b', sublinear_tf=True),
        LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=42))
    model.fit([r['question'] for r in classifier_train], [r['act'] for r in classifier_train])
    assert max(model[-1].n_iter_) < 2000, 'Classifier did not converge'
    joblib.dump(model,out/'act_classifier.joblib')
    save(out/'classifier_parameters.json', {k:str(v) for k,v in model.get_params().items()})
    result = {'corpus_n':len(corpus), 'split_n':{s:len(v) for s,v in data.items()},
              'training_act_counts':dict(Counter(r['act'] for r in classifier_train)),
              'classifier_train_n_before_duplicate_removal':len(data['train']),
              'classifier_train_n_after_duplicate_removal':len(classifier_train),
              'classifier_train_removed_exact_heldout_query_n':len(data['train'])-len(classifier_train),
              'split_audit':audit}
    for split in ('val','test'):
        records = data[split]
        predicted = model.predict([r['question'] for r in records])
        probabilities = model.predict_proba([r['question'] for r in records])
        rows = []
        wrong_rng = {s:np.random.default_rng(s) for s in (7,42,99)}
        for q,pred,prob in zip(records,predicted,probabilities):
            question = q['question']
            sec = parse_section(question)
            base = rank(bm25.scores(tokenize(question)))
            title = rank(title_bm25.scores(tokenize(question)))
            pool = by_section.get(sec,[])
            fallback = not pool
            active = set(pool) if pool else set(range(len(corpus)))
            section_rank = [i for i in base if i in active]
            predicted_rank = [i for i in section_rank if acts[i]==pred]
            rankings = {'paragraph_bm25':base[:10], 'title_paragraph_bm25':title[:10],
                'section_bm25':section_rank[:10],
                'predicted_act_section_hard':predicted_rank[:10],
                'predicted_act_section_empty_fallback':(predicted_rank or section_rank)[:10]}
            wrong = {}
            for seed,rng in wrong_rng.items():
                replacement = None
                rewritten = question
                ranking = base
                if sec is not None:
                    replacement = str(rng.choice([s for s in all_sections if s!=sec]))
                    match = SECTION_RE.search(question)
                    rewritten = question[:match.start(1)] + replacement + question[match.end(1):]
                    wrong_pool = set(by_section[replacement])
                    ranking = [i for i in rank(bm25.scores(tokenize(rewritten))) if i in wrong_pool]
                rankings[f'wrong_section_seed{seed}'] = ranking[:10]
                wrong[str(seed)] = {'section':replacement,'query':rewritten}
            # Gold labels used only below for evaluation and explicit oracle.
            gold = contexts[normalize(q['context'])]
            assert acts[gold]==q['act'], 'Ambiguous context-to-Act gold mapping'
            rankings['oracle_act_section'] = [i for i in section_rank if acts[i]==q['act']][:10]
            row = {'id':q['id'], 'question':question, 'group_id':q['group_id'],
                'gold_index':gold, 'gold_act':q['act'], 'gold_section':sections[gold],
                'parsed_section':sec, 'predicted_act':str(pred),
                'act_probabilities':dict(zip(model.classes_.tolist(),prob.tolist())),
                'zero_residual':not residual_tokens(question), 'section_pool':pool,
                'section_fallback':fallback, 'predicted_act_pool_size':len(predicted_rank),
                'rankings':rankings, 'wrong_sections':wrong,
                'r5':{name:int(gold in ranking[:5]) for name,ranking in rankings.items()}}
            row['r5']['section_random_expected'] = (row['r5']['paragraph_bm25'] if fallback
                         else int(gold in pool)*min(1,5/len(pool)))
            rows.append(row)
        with (out/f'{split}_per_query.jsonl').open('w',encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
        names = list(rows[0]['rankings'])
        metrics = {name:retrieval_metrics([r['rankings'][name] for r in rows],
                                        [r['gold_index'] for r in rows]) for name in names}
        expected = {name:{} for name in ('section_random_expected','size_only_random_expected')}
        for k in (1,3,5,10):
            for name in expected:
                values = []
                for r in rows:
                    pool = r['section_pool']
                    if r['section_fallback']:
                        value = int(r['gold_index'] in r['rankings']['paragraph_bm25'][:k])
                    elif name == 'section_random_expected':
                        value = int(r['gold_index'] in pool)*min(1,k/len(pool))
                    else:
                        value = min(k,len(pool))/len(corpus)
                    values.append(value)
                expected[name][f'recall@{k}'] = float(100*np.mean(values))
        rng = np.random.default_rng(42)
        sims = np.zeros(10000)
        for r in rows:
            if r['section_fallback']:
                sims += r['r5']['paragraph_bm25']
            elif r['gold_index'] in r['section_pool']:
                sims += rng.integers(len(r['section_pool']),size=10000) < 5
        sims *= 100/len(rows)
        truth = [r['gold_act'] for r in rows]
        per_split = {'retrieval':metrics, **expected,
            'section_random_simulation':{'iterations':10000,'seed':42,
                'mean':float(sims.mean()),'sd':float(sims.std()),
                'random_order_95_range':np.quantile(sims,[.025,.975]).tolist()},
            'act_classifier':{'accuracy_pct':100*accuracy_score(truth,predicted),
                'macro_f1_pct':100*f1_score(truth,predicted,average='macro'),
                'class_order':model.classes_.tolist(),
                'confusion_matrix':confusion_matrix(truth,predicted,labels=model.classes_).tolist()},
            'predicted_act_empty_pool_n':sum(r['predicted_act_pool_size']==0 for r in rows),
            'section_fallback_n':sum(r['section_fallback'] for r in rows),
            'candidate_access':{'section_n':sum(r['gold_index'] in r['section_pool'] or r['section_fallback'] for r in rows),
                'predicted_act_section_n':sum((r['gold_index'] in r['section_pool'] or r['section_fallback']) and r['predicted_act']==r['gold_act'] for r in rows)},
            'wrong_equals_gold_section_n':{str(s):sum(r['wrong_sections'][str(s)]['section']==r['gold_section'] for r in rows) for s in (7,42,99)},
            'paired_comparisons':{b+'_minus_'+a:bootstrap(rows,a,b) for a,b in [
                ('section_random_expected','section_bm25'),
                ('section_bm25','predicted_act_section_hard'),
                ('section_bm25','predicted_act_section_empty_fallback')]},
            'subgroups':{}}
        for group in ('zero_residual','content_bearing',*sorted(set(truth))):
            subset = [r for r in rows if (r['zero_residual'] if group=='zero_residual'
                      else not r['zero_residual'] if group=='content_bearing' else r['gold_act']==group)]
            per_split['subgroups'][group] = {'n':len(subset),
                'act_accuracy_pct':100*np.mean([r['gold_act']==r['predicted_act'] for r in subset]),
                'recall@5':{name:100*np.mean([r['r5'][name] for r in subset])
                            for name in (*names,'section_random_expected')}}
        result[split] = per_split
        save(out/'summary.json',result)
        print(split, json.dumps({name:m['recall@5'] for name,m in metrics.items()}),flush=True)
    assert round(result['test']['retrieval']['paragraph_bm25']['recall@5'],2)==22.67
    assert round(result['test']['retrieval']['title_paragraph_bm25']['recall@5'],2)==56.98
    assert round(result['test']['retrieval']['section_bm25']['recall@5'],2)==86.34
    verify(out)
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                    output_hashes={p.name:sha(p) for p in out.iterdir() if p.name!='manifest.json'})
    save(out/'manifest.json',manifest)


if __name__ == '__main__':
    main()
