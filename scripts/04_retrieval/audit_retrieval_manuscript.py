"""Read-only numerical audit; writes only its own audit report (no neural inference)."""
import csv
import json
import re
import statistics
import string
from collections import Counter, defaultdict
from pathlib import Path

import retrieval_diagnostic_controls as controls
from citation_specified_questions import residual_tokens

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/experiments/retrieval'
checks = []


def check(name, actual, expected, tolerance=0.00011):
    passed = abs(actual - expected) <= tolerance
    checks.append(dict(name=name, actual=actual, expected=expected, passed=passed))


def load(name):
    return json.loads((BASE / name).read_text(encoding='utf-8'))


def score(prediction, answer):
    def norm(value):
        return ' '.join(value.lower().translate(str.maketrans('', '', string.punctuation)).split())
    p, a = norm(prediction), norm(answer)
    pt, at = p.split(), a.split()
    overlap = sum((Counter(pt) & Counter(at)).values())
    return float(p == a), 2 * overlap / (len(pt) + len(at)) if pt and at else 0.0


def main():
    test = controls.load_jsonl(controls.TEST_FILE)
    corpus = controls.load_jsonl(controls.CORPUS_FILE)
    predictions = {}
    paths = list((BASE / 'runs/citation_aware_v1').glob('*predictions.jsonl'))
    paths += list((BASE / 'runs/metadata_matched_end_to_end_v1').glob('*predictions.jsonl'))
    paths += list((BASE / 'runs/title_neural_end_to_end_v1').glob('*predictions.jsonl'))
    stages = {}
    for path in paths:
        rows = controls.load_jsonl(path)
        predictions[path.stem] = rows
        check(path.stem + '/questions', len(rows), 344)
        for row in rows:
            em, f1 = score(row['prediction'], row['gold_answer'])
            check(path.stem + '/' + row['id'] + '/em', em, row['em'])
            check(path.stem + '/' + row['id'] + '/f1', f1, row['f1'])
        stages[path.stem] = controls.stage_summary(rows)
        stages[path.stem]['f1_pct'] = 100 * statistics.fmean(r['f1'] for r in rows)
        stages[path.stem]['em_pct'] = 100 * statistics.fmean(r['em'] for r in rows)

    paragraphs = [controls.normalize(r['context']) for r in corpus]
    by_context = {}
    for i, paragraph in enumerate(paragraphs):
        by_context.setdefault(paragraph, i)
    gold = [by_context[controls.normalize(r['context'])] for r in test]
    by_section = defaultdict(list)
    for i, row in enumerate(corpus):
        by_section[controls.parse_section(row['title'])].append(i)
    models = {
        'paragraph_only': controls.BM25([controls.tokenize(p) for p in paragraphs]),
        'title_plus_paragraph': controls.BM25([controls.tokenize(r['title'] + ' ' + p) for r, p in zip(corpus, paragraphs)]),
    }
    rankings = defaultdict(list)
    random_hits = []
    for qi, row in enumerate(test):
        scores = {name: model.scores(controls.tokenize(row['question'])) for name, model in models.items()}
        for name, values in scores.items():
            rankings[name].append(controls.rank(values))
        section = controls.parse_section(row['question'])
        candidates = by_section.get(section, []) if section else []
        for name, model in [('section_filter_plus_bm25', 'paragraph_only'), ('section_filter_plus_title_paragraph_bm25', 'title_plus_paragraph')]:
            rankings[name].append(sorted(candidates, key=lambda i: (-scores[model][i], i)) if candidates else rankings[model][-1])
        boosted = scores['paragraph_only'].copy()
        for i in candidates:
            boosted[i] += 1000
        rankings['section_boost_plus_bm25'].append(controls.rank(boosted))
        random_hits.append(min(1, 5 / len(candidates)) if gold[qi] in candidates else (float(gold[qi] in rankings['paragraph_only'][-1][:5]) if not candidates else 0))
    recomputed_retrieval = {}
    saved = load('retrieval_diagnostic_controls.json')['retrieval_controls']
    for name, rows in rankings.items():
        recomputed_retrieval[name] = controls.retrieval_metrics(rows, gold)
        for metric, value in recomputed_retrieval[name].items():
            check(name + '/' + metric, value, saved[name][metric])
    random_expected = 100 * statistics.fmean(random_hits)
    check('random order exact expected R@5', random_expected, 66.7248)

    cached = load('runs/cross_encoder_rerank_v1/rankings_cache.json')
    cached_metrics = {}
    for name, entry in cached['cache'].items():
        offset = 0
        reranked = []
        for candidates in entry['candidates']:
            scores = entry['scores'][offset:offset + len(candidates)]
            reranked.append([candidates[i] for i in sorted(range(len(candidates)), key=lambda i: -scores[i])])
            offset += len(candidates)
        cached_metrics[name] = {'first_stage': controls.retrieval_metrics(entry['candidates'], cached['gold']),
                                'reranked': controls.retrieval_metrics(reranked, cached['gold'])}
    uk_rows = list(csv.DictReader((BASE / 'runs/uk_statutecorpus_external_v1/per_query.csv').open(encoding='utf-8')))
    uk_summary = load('runs/uk_statutecorpus_external_v1/summary.json')
    modern_indic = load('runs/modern_bge_extension_v1/indic/summary.json')
    modern_uk = load('runs/modern_bge_extension_v1/uk/summary.json')
    metadata_neural = load('runs/metadata_neural_selector_v1/summary.json')
    title_neural = load('runs/title_neural_end_to_end_v1/summary.json')
    for system, metrics in uk_summary['systems'].items():
        rows = [r for r in uk_rows if r['system'] == system]
        check('UK/' + system + '/questions', len(rows), 100)
        for metric, saved_metric in metrics.items():
            check('UK/' + system + '/' + metric, statistics.fmean(float(r[metric]) for r in rows), saved_metric['mean'])

    groups = {r['id']: r['group_id'] for r in test}
    bootstrap = {}
    filtered = predictions['bm25_section_filter_top5_e2_predictions']
    for baseline in ['bm25_top5_e2_predictions', 'title_plus_paragraph_bm25_top5_e2_predictions']:
        bootstrap[baseline] = controls.cluster_bootstrap(predictions[baseline], filtered, groups, 'f1')
    section_bge_rows = controls.load_jsonl(
        BASE / 'runs/metadata_neural_selector_v1/section_filter_bge_selector_predictions.jsonl')
    title_rank1_rows = predictions['title_neural_rank1_selector_predictions']
    bootstrap['title_rank1_vs_section_bge'] = controls.cluster_bootstrap(
        title_rank1_rows, section_bge_rows, groups, 'f1')
    title_section_comparison = bootstrap['title_rank1_vs_section_bge']
    check('title rank1 vs section BGE/F1 difference',
          title_section_comparison['difference_points'], 3.8506)
    check('title rank1 vs section BGE/F1 CI lower',
          title_section_comparison['cluster_bootstrap_95_ci'][0], -0.1602)
    check('title rank1 vs section BGE/F1 CI upper',
          title_section_comparison['cluster_bootstrap_95_ci'][1], 8.7259)
    queries = controls.load_jsonl(BASE / 'citation_ablation/citation_ablation_queries.jsonl')
    no_section = [r for r in queries if not r['eligible']]
    changed = [r for r in no_section if r['original'] != r['content']]
    ablation = load('citation_ablation/citation_contribution_ablation.json')
    ablation_recomputed = {}
    for group in ['citation-template-only', 'lexically-content-bearing']:
        indices = [i for i, r in enumerate(queries) if r['eligible'] and (len(residual_tokens(test[i]['question'])) == 0) == (group == 'citation-template-only')]
        ablation_recomputed[group] = {}
        for variant, field in [('original', 'original'), ('content', 'content'), ('citation', 'citation'), ('wrong', 'wrong_seed42')]:
            hits = {'BM25': [], 'Citation-aware': []}
            for qi in indices:
                query = queries[qi][field]
                scores = models['paragraph_only'].scores(controls.tokenize(query))
                hits['BM25'].append(gold[qi] in controls.rank(scores)[:5])
                section = controls.parse_section(query)
                for i in by_section.get(section, []) if section else []:
                    scores[i] += 1000
                hits['Citation-aware'].append(gold[qi] in controls.rank(scores)[:5])
            ablation_recomputed[group][variant] = {system: 100 * statistics.fmean(values) for system, values in hits.items()}
            for system, value in ablation_recomputed[group][variant].items():
                artifact_group = group if group == 'citation-template-only' else group + ' (with citation)'
                check('ablation/' + group + '/' + system + '/' + variant, round(value, 2), ablation[artifact_group][system][variant]['R@5'], 0.0051)
    tex = (ROOT / 'papers/retrieval_reference/retrieval-reference.tex').read_text(encoding='utf-8')
    tex = re.sub(r'(?<!\\)%.*', '', tex)
    bib = (ROOT / 'papers/retrieval_reference/retrieval-references.bib').read_text(encoding='utf-8')
    keys = set(re.findall(r'@\w+\s*\{\s*([^,]+),', bib))
    cites = {key.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}', tex) for key in group.split(',')}
    missing = sorted(cites - keys)
    check('unresolved citation keys', len(missing), 0)
    def table_rows(label):
        block = next(b for b in re.findall(r'\\begin\{table\}.*?\\end\{table\}', tex, re.S) if '\\label{' + label + '}' in b)
        tabular = re.search(r'\\begin\{tabular\}(.*?)\\end\{tabular\}', block, re.S).group(1)
        result = []
        for line in tabular.splitlines():
            if '&' not in line:
                continue
            values = []
            for cell in line.split('&')[1:]:
                cell = cell.replace('\\\\', '').strip()
                if re.fullmatch(r'-?\d+(?:\.\d+)?', cell):
                    values.append(float(cell))
            if values:
                result.append(values)
        return result

    def compare_table(label, expected):
        actual = table_rows(label)
        check(label + '/row count', len(actual), len(expected))
        for ri, (row, reference) in enumerate(zip(actual, expected)):
            check(f'{label}/{ri}/column count', len(row), len(reference))
            for ci, (value, saved_value) in enumerate(zip(row, reference)):
                check(f'{label}/{ri}/{ci}', value, round(saved_value, 2), 0.00001)

    metric_order = ['recall@1', 'recall@3', 'recall@5', 'recall@10', 'mrr@10']
    review = load('review_robustness_20260920/summary.json')
    decoder = review['reader']
    compare_table('tab:end_to_end_retrieval_qa', [[decoder[name][k] for k in ['gold_available_pct', 'em', 'f1']] for name in ['Gold/raw','BM25top1/raw','BM25top3/raw','BM25/raw']])
    compare_table('tab:prediction_lengths', [[review['saved'][name][k] for k in ['mean_prediction_words','max_prediction_words','predictions_over_100_words']] for name in ['Gold','BM25','Title','Filter']])
    compare_table('tab:answer_length_scores', [[row[k] for k in ['n','em','f1','source']] for name in ['Gold','BM25','Title','Filter'] for row in review['saved'][name]['length_groups'].values()])
    compare_table('tab:parser_disagreements', [[int(r['parsed_first']),int(r['annotated_section'])] for r in review['parser']['disagreements']])
    compare_table('tab:decoder_sensitivity', [[decoder[name+'/'+method]['selected_annotated_source_pct'],decoder[name+'/'+method]['failure_stage_counts']['gold_available_reader_selected_other'],decoder[name+'/'+method]['f1'],decoder[name+'/'+method]['annotation_grounded_f1_pct']] for name in ['BM25','Title','Filter'] for method in ['raw','softmax','rank1','cap128']])
    for name in ['Gold','BM25','Title','Filter']:
        for metric in ['reproduction_source_differences','reproduction_normalized_text_differences']:
            check('decoder reproduction/'+name+'/'+metric,len(decoder[name+'/raw'][metric]),0)
    for path in (BASE/'review_robustness_20260920').glob('*_predictions.json'):
        rows = json.loads(path.read_text(encoding='utf8'))
        check(path.stem+'/count',len(rows),344)
        for row, gold_row in zip(rows,test):
            check(path.stem+'/'+row['id']+'/id',int(row['id']==gold_row['id']),1)
            em,f1=score(row['prediction'],gold_row['answer'])
            check(path.stem+'/'+row['id']+'/em',em,row['em'])
            check(path.stem+'/'+row['id']+'/f1',f1,row['f1'])
    compare_table('tab:dense_retrieval', [[r[k] for k in metric_order] for r in load('dense_retrieval_report.json')['results'].values()])
    compare_table('tab:rerank', [[r[k] for k in metric_order[:-1] + ['recall@50', 'mrr@10']] for r in load('cross_encoder_rerank_report.json')['results'].values()])
    modern_metric_order = ['recall@1', 'recall@3', 'recall@5', 'recall@10',
                           'recall@50', 'mrr@10', 'ndcg@10']
    compare_table('tab:modern_bge_indic',
                  [[r[k] for k in modern_metric_order]
                   for r in modern_indic['results'].values()] +
                  [[r[k] for k in modern_metric_order]
                   for r in metadata_neural['retrieval_results'].values()])
    for name, values in cached_metrics.items():
        for kind, metrics in values.items():
            system = name + (' + cross-encoder' if kind == 'reranked' else '')
            for metric, value in metrics.items():
                check('cached ranking/' + system + '/' + metric, round(value, 2), load('cross_encoder_rerank_report.json')['results'][system][metric])
    names = ['gold_paragraph_top5_e2_predictions', 'bm25_top5_e2_predictions', 'title_plus_paragraph_bm25_top5_e2_predictions', 'bm25_section_boost_top5_e2_predictions', 'bm25_section_filter_top5_e2_predictions']
    selector = metadata_neural['independent_source_selector']
    pipeline_keys = ['gold_available_pct', 'selected_annotated_source_pct',
                     'em_pct', 'f1_pct', 'annotation_grounded_f1_pct']
    title_reader = 'title_neural_top5_reader_selector_predictions'
    title_rank1 = 'title_neural_rank1_selector_predictions'
    compare_table('tab:citation_aware_end_to_end',
                  [[stages[n][k] for k in pipeline_keys] for n in names[:3]] +
                  [[stages[n][k] for k in pipeline_keys]
                   for n in [title_reader, title_rank1]] +
                  [[stages[n][k] for k in pipeline_keys] for n in names[3:]] +
                  [[selector[k] for k in ['retrieval_recall@5',
                                           'selected_annotated_source_pct',
                                           'em', 'f1',
                                           'annotation_grounded_f1']]])
    for saved_name, prediction_name in [
        ('reader_selector_over_top5', title_reader),
        ('reranker_rank1_selector', title_rank1),
    ]:
        saved_title = title_neural[saved_name]
        calculated = stages[prediction_name]
        for saved_key, calculated_key in [
            ('retrieval_recall@5_pct', 'gold_available_pct'),
            ('selected_annotated_source_pct', 'selected_annotated_source_pct'),
            ('conditional_source_selection_pct',
             'conditional_source_selection_pct_given_available'),
            ('em_pct', 'em_pct'), ('f1_pct', 'f1_pct'),
            ('annotation_grounded_f1_pct', 'annotation_grounded_f1_pct'),
        ]:
            check('title neural/' + saved_name + '/' + saved_key,
                  calculated[calculated_key], saved_title[saved_key])
    selector_counts = selector['stage_counts']
    stage_names = [
        'bm25_top5_e2_predictions',
        'title_plus_paragraph_bm25_top5_e2_predictions',
        title_reader,
        title_rank1,
        'bm25_section_boost_top5_e2_predictions',
        'bm25_section_filter_top5_e2_predictions',
    ]
    compare_table('tab:stage_decomposition',
                  [[stages[n]['failure_stage_counts'][k] for k in [
                      'gold_absent_from_top5',
                      'gold_available_reader_selected_other',
                      'gold_selected_non_em_extraction',
                      'gold_selected_normalized_em',
                  ]] + [stages[n]['conditional_source_selection_pct_given_available'],
                        stages[n]['annotation_grounded_f1_pct']]
                   for n in stage_names] +
                  [[selector_counts[k] for k in ['source_absent', 'available_other_selected', 'annotated_source_selected_non_em', 'annotated_source_selected_em']] + [selector['conditional_source_selection_pct'], selector['annotation_grounded_f1']]])
    # The current table combines the original five systems with the four new
    # BGE systems (BM25 is shared and appears only once).
    uk_metric_order = ['recall@5', 'recall@10', 'recall@50', 'mrr@10',
                       'ndcg@10', 'grade3_recall@10']
    uk_expected = [[r[k]['mean'] * 100 for k in uk_metric_order]
                   for r in uk_summary['systems'].values()]
    uk_expected += [[r[k]['mean'] * 100 for k in uk_metric_order]
                    for name, r in modern_uk['systems'].items() if name != 'bm25']
    compare_table('tab:uk_external_retrieval', uk_expected)
    subgroups = load('runs/uk_reference_followup_v1/summary.json')['G2_existing_ranking_subgroups']
    compare_table('tab:uk_reference_subgroups', [[subgroups[g][s][m] for g in ['supported_resolved_reference', 'no_recognised_reference'] for m in ['recall@5_pct', 'ndcg@10_pct']] for s in uk_summary['systems']])
    qa = load('retrieval_qa_report.json')
    compare_table('tab:citation_specified', [[r[k] for k in ['n', 'gold_em_pct', 'gold_f1_pct', 'bm25_top5_em_pct', 'bm25_top5_f1_pct']] for r in qa['question_type_results'].values()] + [[344, qa['results'][0]['em_pct'], qa['results'][0]['f1_pct'], qa['results'][3]['em_pct'], qa['results'][3]['f1_pct']]])
    robustness = load('citation_template_classifier_robustness.json')
    compare_table('tab:classifier_continuous', [[r['n'], r['bm25_end_to_end_f1_pct']] for r in robustness['continuous_residual_token_analysis']])
    with (BASE / 'citation_template_classifier_audit.csv').open(encoding='utf-8-sig', newline='') as stream:
        classifier_rows = list(csv.DictReader(stream))
    within_act_expected = []
    for act in ['copyright_act', 'gi_act', 'patent_act', 'semiconductor_layout_design_act', 'trademark_act']:
        values = []
        for question_class in ['citation-template-only', 'lexically-content-bearing']:
            subset = [row for row in classifier_rows
                      if row['act'] == act and row['assigned_class'] == question_class]
            values.extend([
                len(subset),
                100 * statistics.fmean(float(row['bm25_recall_at_5']) for row in subset),
                statistics.fmean(float(row['bm25_end_to_end_f1_pct']) for row in subset),
            ])
        within_act_expected.append(values)
    compare_table('tab:within_act_question_groups', within_act_expected)
    dense = load('dense_retrieval_report.json')['results']
    section_results = load('citation_aware_retrieval_report.json')['results']
    rerank_results = load('cross_encoder_rerank_report.json')['results']
    compare_table('tab:citation_aware',
                  [[r[k] for k in metric_order] for r in [
                      modern_indic['results']['bm25'],
                      modern_indic['results']['rrf_bm25_bge_m3'],
                      modern_indic['results']['rrf_bm25_bge_m3_bge_reranker_v2_m3'],
                      section_results['BM25 + section boost']]])
    ablation_table = []
    for group in ['citation-template-only', 'lexically-content-bearing (with citation)']:
        for system in ['BM25', 'Hybrid+CE', 'Citation-aware']:
            ablation_table.append([ablation[group][system][variant]['R@5'] for variant in ['original', 'content', 'citation', 'wrong']])
    ablation_table += [[63.16, 63.16], [63.16, 63.16]]
    compare_table('tab:citation_ablation', ablation_table)
    for threshold in robustness['threshold_sensitivity']:
        for group, saved_group in threshold['groups'].items():
            ids = {r['id'] for r in test if (len(residual_tokens(r['question'])) <= threshold['max_residual_tokens_for_citation_template_only']) == (group == 'citation-template-only')}
            check(f'classifier/{threshold["max_residual_tokens_for_citation_template_only"]}/{group}/n', len(ids), saved_group['n'])
            for prediction_name, metric in [('gold_paragraph_top5_e2_predictions', 'gold_paragraph_f1_pct'), ('bm25_top5_e2_predictions', 'bm25_end_to_end_f1_pct')]:
                value = 100 * statistics.fmean(r['f1'] for r in predictions[prediction_name] if r['id'] in ids)
                check(f'classifier/{threshold["max_residual_tokens_for_citation_template_only"]}/{group}/{metric}', value, saved_group[metric])
    act_rows = list(csv.DictReader(next((ROOT / 'benchmark_analysis/outputs').glob('difficulty_scores_E2*.csv')).open(encoding='utf-8')))
    overlap = [float(r['Q-Ctx overlap']) for r in act_rows]
    reader_f1 = [float(r['Test F1']) for r in act_rows]
    # Additional coverage for mixed-format tables omitted by compare_table.
    act_map = [('Plant Varieties', 'Plant Var.', 'plant_varieties_farmers_rights_act'),
               ('Copyright', 'Copyright', 'copyright_act'), ('GI', 'GI', 'gi_act'),
               ('Patents', 'Patents', 'patent_act'), ('Trade Marks', 'Trade Marks', 'trademark_act'),
               ('Semiconductor', 'Semicond.', 'semiconductor_layout_design_act'),
               ('Designs', 'Designs', 'design_act')]
    act_lookup = {r['Act']: r for r in act_rows}
    dense_acts = load('dense_retrieval_report.json')['per_act_recall_at_5']
    ce_acts = load('cross_encoder_rerank_report.json')['per_act_recall5_best']['values']
    actual_acts = table_rows('tab:retrieval_per_act')
    for values, (label, reader_label, act) in zip(actual_acts, act_map):
        source = act_lookup[reader_label]
        expected = [float(source['Q-Ctx overlap']), dense_acts['BM25'][act],
                    list(dense_acts.values())[1][act], ce_acts[act], round(float(source['Test F1']), 2)]
        for i, (value, reference) in enumerate(zip(values, expected)):
            check(f'tab:retrieval_per_act/{label}/{i}', value, reference)
    threshold_block = next(b for b in re.findall(r'\\begin\{table\}.*?\\end\{table\}', tex, re.S)
                           if '\\label{tab:classifier_thresholds}' in b)
    for row, saved_threshold in zip([line for line in threshold_block.splitlines() if re.match(r'\s*\d', line)], robustness['threshold_sensitivity']):
        actual = [float(v) for v in re.findall(r'\d+(?:\.\d+)?', row)]
        groups = list(saved_threshold['groups'].values())
        expected = [saved_threshold['max_residual_tokens_for_citation_template_only'],
                    groups[0]['n'], groups[1]['n']]
        expected += [round(g[m], 2) for m in ['gold_paragraph_f1_pct', 'bm25_end_to_end_f1_pct'] for g in groups]
        for i, (value, reference) in enumerate(zip(actual, expected)):
            check(f'tab:classifier_thresholds/{actual[0]}/{i}', value, reference)
    examples = [('Under Section 14, what is stated in clause (b)?', []),
                ('What does Section 30A provide regarding application of section 19?', ['application']),
                ('Under Section 33 (Effect of acquiescence), what is provided in sub-section (1)?', ['effect', 'acquiescence'])]
    for question, expected in examples:
        check('classifier example/' + question, int(sorted(residual_tokens(question)) == sorted(expected)), 1)
    loo = {r['Act']: statistics.correlation(overlap[:i] + overlap[i+1:], reader_f1[:i] + reader_f1[i+1:]) for i, r in enumerate(act_rows)}
    check('reader-overlap full correlation', round(statistics.correlation(overlap, reader_f1), 3), -0.785)
    check('reader-overlap LOO minimum', round(min(loo.values()), 3), -0.907)
    check('reader-overlap LOO maximum', round(max(loo.values()), 3), -0.674)
    check('reader-overlap LOO excludes Plant Varieties', round(loo['Plant Var.'], 3), -0.674)
    report = dict(checks=checks, failed_checks=[c for c in checks if not c['passed']],
                  recomputed_reader_metrics=stages, recomputed_retrieval=recomputed_retrieval,
                  cached_reranking_metrics=cached_metrics, random_expected_recall5=random_expected,
                  recomputed_sparse_ablation=ablation_recomputed,
                  reader_overlap_leave_one_act_out=loo,
                  recomputed_cluster_bootstrap=bootstrap,
                  no_section_queries=len(no_section), no_section_changed_by_cleanup=len(changed),
                  missing_citation_keys=missing,
                  scope='Saved answer scores, sparse retrieval, cached reranking, UK per-query means, paired bootstrap and citation-key integrity. No model retraining or external publication verification.')
    output = BASE / 'manuscript_audit/summary.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['failed_checks', 'recomputed_cluster_bootstrap', 'no_section_queries', 'no_section_changed_by_cleanup']}, indent=2))
    print(f'{len(checks)} checks; report: {output}')
    if report['failed_checks']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
