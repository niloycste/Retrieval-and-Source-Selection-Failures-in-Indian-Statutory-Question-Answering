"""Check local citation keys, literal DOI/URL fields and compiled PDF links.

This is a structural/link-integrity check, not publisher-metadata verification.
Run after BibTeX and the final PDFLaTeX passes. Writes only its own report.
"""
import json
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'papers/retrieval_reference'


def main():
    bib = (PAPER / 'retrieval-references.bib').read_text(encoding='utf-8')
    tex = (PAPER / 'retrieval-reference.tex').read_text(encoding='utf-8')
    tex = re.sub(r'(?m)(?<!\\)%.*$', '', tex)
    keys = re.findall(r'(?m)^@\w+\s*\{\s*([^,\s]+)\s*,', bib)
    citations = {
        key.strip()
        for group in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}', tex)
        for key in group.split(',')
    }
    fields = {name: re.findall(r'(?im)^\s*' + name + r'\s*=\s*\{([^}]+)\}', bib)
              for name in ('doi', 'url')}
    problems = []
    for key, n in Counter(keys).items():
        if n > 1:
            problems.append(f'Duplicate bibliography key: {key}')
    for key in sorted(citations - set(keys)):
        problems.append(f'Missing bibliography key: {key}')
    for name, values in fields.items():
        for value in values:
            if re.search(r'\s', value):
                problems.append(f'Whitespace in {name}: {value!r}')
            if name == 'doi' and not re.fullmatch(r'10\.\d{4,9}/\S+', value):
                problems.append(f'Unexpected DOI syntax: {value}')
            if name == 'url' and not value.startswith(('https://', 'http://')):
                problems.append(f'Unexpected URL scheme: {value}')
    pdf = PdfReader(PAPER / 'retrieval-reference.pdf')
    links = []
    for page_number, page in enumerate(pdf.pages, 1):
        for ref in page.get('/Annots', []):
            annotation = ref.get_object()
            action = annotation.get('/A')
            if action is not None:
                action = action.get_object()
                if '/URI' in action:
                    links.append({'page': page_number, 'uri': str(action['/URI'])})
    targets = {item['uri'] for item in links}
    for uri in sorted(targets):
        if re.search(r'\s', uri):
            problems.append(f'Whitespace in PDF link target: {uri!r}')
    expected = set(fields['url']) | {'https://doi.org/' + doi for doi in fields['doi']}
    # All entries in this standalone bibliography should be cited and rendered.
    for key in sorted(set(keys) - citations):
        problems.append(f'Uncited bibliography entry: {key}')
    for uri in sorted(expected - targets):
        problems.append(f'Bibliography URL/DOI missing from PDF link targets: {uri}')
    report = dict(
        scope='Local structural and PDF-link integrity only; not metadata verification',
        bibliography_entries=len(keys), cited_keys=len(citations),
        doi_fields=len(fields['doi']), url_fields=len(fields['url']),
        pdf_pages=len(pdf.pages), pdf_uri_annotations=len(links),
        unique_pdf_uri_targets=len(targets), problems=problems,
        bibliography_targets=sorted(expected), pdf_links=links,
    )
    output = ROOT / 'results/experiments/retrieval/manuscript_audit/bibliography_links.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ('bibliography_targets', 'pdf_links')}, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
