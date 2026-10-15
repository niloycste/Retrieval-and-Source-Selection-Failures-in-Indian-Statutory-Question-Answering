"""Post-review checks; writes only a new audit directory, never legacy results.

Default: deterministic ranking/parser and saved-output checks.
--with-reader: frozen-reader inference for decoder/length sensitivity. No training.
The optional inference cache is resumable and records per-question/candidate
outputs, not merely the winning paragraph. Human annotations are not generated.
"""
import argparse
import hashlib
import json
import os
import statistics as st
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import retrieval_diagnostic_controls as d
from citation_specified_questions import INTERROGATIVE_FRAME

OUT = ROOT/'results/experiments/retrieval/review_robustness_20260920'
MODEL = ROOT/'artifacts/models/squad_v1/baseline_e2_squadv1/final'


def write(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf8')


def mean(values):
    return 100*st.mean(values) if values else None


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--with-reader', action='store_true')
    opt = args.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    corpus, tests = d.load_jsonl(d.CORPUS_FILE), d.load_jsonl(d.TEST_FILE)
    contexts = [d.normalize(r['context']) for r in corpus]
    by_context = {}
    for i,c in enumerate(contexts):
        by_context.setdefault(c,i)
    gold = [by_context[d.normalize(r['context'])] for r in tests]
    sections = [d.parse_section(r.get('title','')) for r in corpus]
    bm = d.BM25([d.tokenize(c) for c in contexts])
    title = d.BM25([d.tokenize(r['title']+' '+c) for r,c in zip(corpus,contexts)])
    rankings = {key:[] for key in ('BM25','Title','Filter','Gold')}
    parser_rows = []
    for i,t in enumerate(tests):
        scores = bm.scores(d.tokenize(t['question']))
        base = d.rank(scores)
        target = d.parse_section(t['question'])
        matched = [j for j,s in enumerate(sections) if target is not None and s == target]
        filtered = sorted(matched, key=lambda j:(-scores[j],j)) if matched else base
        rankings['BM25'].append(base[:5])
        rankings['Title'].append(d.rank(title.scores(d.tokenize(t['question'])))[:5])
        rankings['Filter'].append(filtered[:5])
        rankings['Gold'].append([gold[i]])
        all_refs = [m.upper() for m in d.SECTION_RE.findall(t['question'])]
        parser_rows.append(dict(id=t['id'], question=t['question'], parsed_first=target,
            all_regex_matches=all_refs, annotated_section=sections[gold[i]],
            agrees_with_annotation=target==sections[gold[i]] if target else None,
            matching_passages=len(matched), matching_acts=len({corpus[j]['act'] for j in matched}),
            human_detection_judgment=None, human_target_judgment=None, human_identifiability_judgment=None))
    write('parser_items.json',parser_rows)
    parsed = [r for r in parser_rows if r['parsed_first']]
    report = dict(scope='Post-review automatic diagnostics; not human validation',
        parser=dict(regex=d.SECTION_RE.pattern, parsed=len(parsed),
            first_agrees=sum(r['agrees_with_annotation'] for r in parsed),
            multiple_matches=sum(len(r['all_regex_matches'])>1 for r in parsed),
            multiple_distinct=sum(len(set(r['all_regex_matches']))>1 for r in parsed),
            disagreements=[r for r in parsed if not r['agrees_with_annotation']]),
        frame_lexicon=sorted(INTERROGATIVE_FRAME),retrieval={},saved={},source_rank1={})
    saved_paths = {'BM25':d.PREDICTION_DIR/'bm25_top5_e2_predictions.jsonl',
        'Filter':d.PREDICTION_DIR/'bm25_section_filter_top5_e2_predictions.jsonl',
        'Gold':d.PREDICTION_DIR/'gold_paragraph_top5_e2_predictions.jsonl',
        'Title':ROOT/'results/experiments/retrieval/runs/metadata_matched_end_to_end_v1/title_paragraph_bm25_top5_e2_predictions.jsonl'}
    # Discover the metadata control filename, while requiring an unambiguous file.
    title_paths=list(saved_paths['Title'].parent.glob('*predictions.jsonl'))
    assert len(title_paths)==1
    saved_paths['Title']=title_paths[0]
    saved={key:d.load_jsonl(path) for key,path in saved_paths.items()}
    for key,ranks in rankings.items():
        report['retrieval'][key]=d.retrieval_metrics(ranks,gold)
        available=[i for i,r in enumerate(ranks) if gold[i] in r]
        report['source_rank1'][key]=dict(selected_gold=sum(r[0]==g for r,g in zip(ranks,gold)),
            available_other=sum(ranks[i][0]!=gold[i] for i in available),available=len(available))
        rows=saved[key]
        assert [r['id'] for r in rows]==[r['id'] for r in tests]
        lengths=[len(r['prediction'].split()) for r in rows]
        groups={}
        for label,lo,hi in [('1--10',1,10),('11--30',11,30),('31+',31,100000)]:
            subgroup=[r for r,t in zip(rows,tests) if lo<=len(t['answer'].split())<=hi]
            groups[label]=dict(n=len(subgroup),em=mean([r['em'] for r in subgroup]),
                f1=mean([r['f1'] for r in subgroup]),source=mean([r['selected_context_is_gold'] for r in subgroup]))
        report['saved'][key]=dict(mean_prediction_words=st.mean(lengths),max_prediction_words=max(lengths),
            predictions_over_100_words=sum(n>100 for n in lengths),length_groups=groups)
    legacy=d.load_jsonl(ROOT/'results/experiments/retrieval/runs/journal_retrieval_transfer_e2prime_canonical_v1/retrieval_qa/bm25_top_5_qa_predictions.jsonl')
    legacy_contexts=list(dict.fromkeys(str(r['context']).strip() for r in d.load_jsonl(ROOT/'data/processed/indic_ipr_qa/merged_ip_qa.jsonl') if str(r.get('context','')).strip()))
    exact_corpus={str(r['context']).strip():i for i,r in enumerate(corpus)}
    legacy_to_canonical={i:exact_corpus[c] for i,c in enumerate(legacy_contexts)}
    assert len(legacy_contexts)==len(corpus)
    assert len(set(legacy_to_canonical.values()))==len(corpus)
    tie_rows=[]
    for i,(a,b) in enumerate(zip(legacy,saved['BM25'])):
        old=[legacy_to_canonical[int(c['context_id'].split('_')[-1])] for c in a['top_candidates']]
        if old!=rankings['BM25'][i]:
            tie_rows.append(dict(id=a['id'],legacy=old,canonical=rankings['BM25'][i],
                selected_changed=legacy_to_canonical[int(a['selected_context_id'].split('_')[-1])]!=b['selected_context_index'],
                prediction_changed=a['prediction']!=b['prediction'],f1_delta=b['f1']-a['f1']))
    report['legacy_top5_differences']=tie_rows
    previous=json.loads((OUT/'summary.json').read_text(encoding='utf8')) if (OUT/'summary.json').exists() else {}
    if 'reader' in previous:
        report['reader']=previous['reader']
    write('summary.json',report)
    print(json.dumps({k:report[k] for k in ['parser','source_rank1','saved']},indent=2),flush=True)
    if not opt.with_reader:
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    from scripts.qa_evaluation import best_span_from_logits,compute_em_f1
    torch.set_num_threads(4)
    pairs=sorted({(i,j) for rs in rankings.values() for i,r in enumerate(rs) for j in r})
    cache_path=OUT/'reader_candidates.jsonl'
    cache={(r['query'],r['context']):r for r in d.load_jsonl(cache_path)} if cache_path.exists() else {}
    manifest=dict(model=str(MODEL.relative_to(ROOT)),max_tokens=384,stride=128,
        cap_tokens=128,cap_basis='Fixed post-review sensitivity choice, not tuned to test scores',
        softmax='Separate start/end log_softmax over valid context positions within each window; sum; max across windows',
        pairs=len(pairs),threads=4,model_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in MODEL.iterdir() if p.suffix in ('.json','.safetensors')})
    manifest['input_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [d.CORPUS_FILE,d.TEST_FILE]}
    old_manifest=json.loads((OUT/'reader_manifest.json').read_text(encoding='utf8')) if (OUT/'reader_manifest.json').exists() else None
    if cache and old_manifest:
        for key in ['model_files','max_tokens','stride','cap_tokens','softmax']:
            assert old_manifest[key]==manifest[key], f'Cache provenance mismatch: {key}'
        if 'input_sha256' in old_manifest:
            assert old_manifest['input_sha256']==manifest['input_sha256'], 'Cache inputs changed'
    write('reader_manifest.json',manifest)
    missing=[p for p in pairs if p not in cache]
    if missing:
        tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True)
        model=AutoModelForQuestionAnswering.from_pretrained(str(MODEL),local_files_only=True).eval()
    for chunk_start in range(0,len(missing),32):
        chunk=missing[chunk_start:chunk_start+32]
        encoded=tokenizer([tests[i]['question'] for i,j in chunk],[contexts[j] for i,j in chunk],
            return_tensors='pt',truncation='only_second',max_length=384,stride=128,
            return_overflowing_tokens=True,return_offsets_mapping=True,padding=True)
        seqs=[encoded.sequence_ids(i) for i in range(len(encoded['input_ids']))]
        offsets=encoded.pop('offset_mapping').tolist()
        mapping=encoded.pop('overflow_to_sample_mapping').tolist()
        output=[dict(query=i,context=j,windows=mapping.count(k),raw=['',float('-inf')],softmax=['',float('-inf')],cap128=['',float('-inf')]) for k,(i,j) in enumerate(chunk)]
        for start in range(0,len(mapping),8):
            with torch.inference_mode():
                logits=model(**{k:v[start:start+8] for k,v in encoded.items()})
            for loc,feat in enumerate(range(start,min(start+8,len(mapping)))):
                sample=mapping[feat]
                s,e=logits.start_logits[loc],logits.end_logits[loc]
                params=dict(context=contexts[chunk[sample][1]],offset_mapping=offsets[feat],sequence_ids=seqs[feat])
                raw=best_span_from_logits(**params,start_logits=s,end_logits=e)
                cap=best_span_from_logits(**params,start_logits=s,end_logits=e,max_answer_tokens=128)
                positions=[p for p,seq in enumerate(seqs[feat]) if seq==1 and offsets[feat][p][1]>offsets[feat][p][0]]
                soft=(raw[0],raw[1]-float(torch.logsumexp(s[positions],0))-float(torch.logsumexp(e[positions],0)))
                for kind,value in [('raw',raw),('softmax',soft),('cap128',cap)]:
                    if value[1]>output[sample][kind][1]:
                        output[sample][kind]=value
        with cache_path.open('a',encoding='utf8') as handle:
            for row in output:
                handle.write(json.dumps(row,ensure_ascii=False)+'\n')
                cache[(row['query'],row['context'])]=row
        print(f'Cached {len(cache)}/{len(pairs)} question-passage pairs',flush=True)
    results={}
    for system,rs in {**rankings,'BM25top1':[r[:1] for r in rankings['BM25']], 'BM25top3':[r[:3] for r in rankings['BM25']]}.items():
        for method in ['raw','softmax','rank1','cap128']:
            kind='raw' if method=='rank1' else method
            rows=[]
            for i,r in enumerate(rs):
                choices=r[:1] if method=='rank1' else r
                score,pos,j,pred=max((cache[(i,j)][kind][1],pos,j,cache[(i,j)][kind][0]) for pos,j in enumerate(choices,1))
                em,f1=compute_em_f1(pred,tests[i]['answer'])
                rows.append(dict(id=tests[i]['id'],prediction=pred,selected_context_index=j,em=em,f1=f1,
                    selected_context_is_gold=j==gold[i],retrieval_hit_at_k=gold[i] in r,windows=cache[(i,j)]['windows']))
            key=system+'/'+method
            results[key]=dict(d.stage_summary(rows),em=mean([r['em'] for r in rows]),f1=mean([r['f1'] for r in rows]),
                selected_multiwindow=sum(r['windows']>1 for r in rows),
                candidate_multiwindow=sum(cache[(i,j)]['windows']>1 for i,r in enumerate(rs) for j in r),
                candidate_pairs=sum(map(len,rs)))
            if system in saved and method=='raw':
                results[key]['reproduction_differences']=[r['id'] for r,old in zip(rows,saved[system]) if r['prediction']!=old['prediction'] or r['selected_context_index']!=old['selected_context_index']]
                results[key]['reproduction_source_differences']=[r['id'] for r,old in zip(rows,saved[system]) if r['selected_context_index']!=old['selected_context_index']]
                results[key]['reproduction_normalized_text_differences']=[r['id'] for r,old in zip(rows,saved[system]) if d.normalize(r['prediction'])!=d.normalize(old['prediction'])]
            write(system+'_'+method+'_predictions.json',rows)
    report['reader']=results
    write('summary.json',report)
    print('Completed frozen-reader sensitivity analyses.',flush=True)


if __name__=='__main__':
    main()
