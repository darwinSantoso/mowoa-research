"""Rebuild the canonical JSON locally (it holds IndoSum text, so it is not in the repo).

    python tools/rebuild_canonical_json.py --indosum <dir>/test.01.jsonl \
        --ids data/wp1_ids.csv --out /content

Writes the same content under both names the notebooks open:
  mowoa_summaries_150_rc1_w32_update_test.json   (Phase B, Phase D)
  mowoa_summaries_150_rc1_w32.json               (notebook 04)

Lists, one entry per article, rows 0..149 of test.01.jsonl in file order:
  golden_summaries     IndoSum `summary`, joined like tokens_to_narrative()
  source_texts         all native sentences joined by spaces
  generated_summaries  the first three native sentences (the w = 32 run delivered
                       native LEAD-3 in 150/150 articles)
The join rules are copied from the notebooks. WP1 W1 checks the rebuilt file
against IndoSum again.
"""
import argparse
import csv
import json
import os
import sys

NAMES = ['mowoa_summaries_150_rc1_w32_update_test.json', 'mowoa_summaries_150_rc1_w32.json']
K_SUMMARY = 3


# same as split_text_into_sentences() in the notebooks
def split_text_into_sentences(paragraphs):
    sentences = []
    for paragraph in paragraphs:
        for sentence_tokens in paragraph:
            sentence = ' '.join(sentence_tokens)
            sentence = sentence.replace(' ,', ',')
            sentence = sentence.replace(' .', '.')
            sentence = sentence.replace(' :', ':')
            sentence = sentence.replace(' ;', ';')
            sentence = sentence.replace(' !', '!')
            sentence = sentence.replace(' ?', '?')
            sentence = sentence.replace(' )', ')')
            sentence = sentence.replace('( ', '(')
            if sentence.strip():
                sentences.append(sentence.strip())
    return sentences


# same as tokens_to_narrative() in the notebooks (no bracket rules here)
def tokens_to_narrative(tokenized_sentences):
    sentences = []
    for tokens in tokenized_sentences:
        sentence = " ".join(tokens)
        sentence = sentence.replace(" ,", ",").replace(" .", ".")
        sentence = sentence.replace(" :", ":").replace(" ;", ";")
        sentence = sentence.replace(" !", "!").replace(" ?", "?")
        sentences.append(sentence)
    return " ".join(sentences)


def read_ids(path):
    # data/wp1_ids.csv: article,index_in_test01,indosum_id,category,source
    with open(path, encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    col = next((c for c in ('indosum_id', 'id') if c in rows[0]), None)
    if col is None:
        sys.exit(f'{path}: no indosum_id/id column, found {list(rows[0])}')
    if 'index_in_test01' in rows[0]:
        rows.sort(key=lambda r: int(r['index_in_test01']))
    return [r[col].strip() for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--indosum', required=True, help='path to IndoSum test.01.jsonl')
    ap.add_argument('--ids', default='data/wp1_ids.csv', help='the 150 evaluated IDs (data/wp1_ids.csv)')
    ap.add_argument('--out', default='.', help='output folder (the Colab working dir is /content)')
    ap.add_argument('--n', type=int, default=150)
    args = ap.parse_args()

    rows = []
    with open(args.indosum, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            rows.append(json.loads(line))
    if len(rows) < args.n:
        sys.exit(f'{args.indosum} has only {len(rows)} rows, need {args.n}')

    if os.path.exists(args.ids):
        ids = read_ids(args.ids)
        got = [str(r.get('id', '')) for r in rows]
        if ids[:args.n] != got:
            bad = next(i for i, (a, b) in enumerate(zip(ids, got)) if a != b) if len(ids) >= args.n else len(ids)
            sys.exit(f'ID mismatch at row {bad}: {args.ids} has {ids[bad] if bad < len(ids) else "nothing"}, '
                     f'{args.indosum} has {got[bad]}. Wrong file or wrong fold?')
        print(f'IDs match {args.ids} ({args.n}/{args.n})')
    else:
        print(f'warning: {args.ids} not found, IDs not checked')

    golden, source, generated = [], [], []
    for r in rows:
        sents = split_text_into_sentences(r['paragraphs'])
        golden.append(tokens_to_narrative(r['summary']))
        source.append(' '.join(sents))
        generated.append(' '.join(sents[:K_SUMMARY]))

    data = {'generated_summaries': generated, 'golden_summaries': golden, 'source_texts': source}
    os.makedirs(args.out, exist_ok=True)
    for name in NAMES:
        p = os.path.join(args.out, name)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print('wrote', p)
    print('these files contain IndoSum text, keep them out of the repository')


if __name__ == '__main__':
    main()
