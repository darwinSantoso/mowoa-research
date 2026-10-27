"""Pre-publication scan of the repository folder.

    python tools/check_release.py .
    python tools/check_release.py . --indosum data/indosum/test.01.jsonl

Checks
  files     IndoSum text, caches, checkpoints and weights that must not be committed
  size      files over 1 MB
  outputs   notebook outputs, execution counts, Colab/widget metadata
  secrets   API keys and tokens
  links     private Colab / Google Drive links
  personal  local paths (C:\\Users, OneDrive, /Users/...) and e-mail addresses
  module    src/mowoa_mo_shared.py == the %%writefile block in every notebook that has one
  indosum   (with --indosum) 8-token fragments of the evaluated articles or their references

Prints PASS / WARN / FAIL per check; exit code 1 if anything FAILs.
The scan reads the working tree. Run `git status --ignored` after `git add .` as well.
"""
import argparse
import fnmatch
import json
import os
import re
import sys

SKIP_DIRS = {'.git', '__pycache__', '.ipynb_checkpoints', 'original'}
TEXT_EXT = {'.py', '.ipynb', '.md', '.txt', '.csv', '.tsv', '.cff', '.json', '.yml', '.yaml',
            '.toml', '.cfg', '.ini', '.sh', ''}

FORBIDDEN = [
    '*.jsonl', '*.pkl', '*.pt', '*.bin', '*.safetensors', '*.ckpt', '*.h5',
    'mowoa_summaries_150*.json', 'mowoa_plus_summaries_150*.json',
    'all_summaries_150_archive.json', 'woa_singleobj_summaries_150.json',
    'indobert_summaries_150.json', 'selected_sentences_*.tsv', 'wp5_processing.txt',
]
FORBIDDEN_DIRS = ['phaseC', 'phaseD/ladder', 'runs', 'peer_runs', 'report_figures']

SECRETS = [
    (r'hf_[A-Za-z0-9]{30,}', 'Hugging Face token'),
    (r'sk-[A-Za-z0-9_\-]{20,}', 'OpenAI-style key'),
    (r'ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}', 'GitHub token'),
    (r'AKIA[0-9A-Z]{16}', 'AWS access key'),
    (r'AIza[0-9A-Za-z_\-]{35}', 'Google API key'),
    (r'xox[baprs]-[A-Za-z0-9\-]{10,}', 'Slack token'),
    (r'-----BEGIN [A-Z ]*PRIVATE KEY-----', 'private key'),
    (r'(?i)kaggle_key\s*[=:]\s*["\']?[0-9a-f]{20,}', 'Kaggle key'),
    (r'(?i)(api[_-]?key|secret|password|passwd|token)\s*=\s*["\'][^"\']{8,}["\']', 'hard-coded secret'),
]
LINKS = [
    (r'colab\.research\.google\.com/drive/[A-Za-z0-9_\-]+', 'private Colab link'),
    (r'drive\.google\.com/(file|drive|open)[^\s"\']*', 'Google Drive link'),
    (r'docs\.google\.com/[^\s"\']+', 'Google Docs link'),
]
PERSONAL = [
    (r'[A-Za-z]:\\\\?Users\\\\?[^\\\s"\']+', 'Windows user path', 'FAIL'),
    (r'(?<![\w/])/Users/[^/\s"\']+', 'macOS user path', 'FAIL'),
    (r'(?<![\w/])/home/(?!runner)[^/\s"\']+', 'Linux home path', 'FAIL'),
    (r'OneDrive', 'OneDrive path', 'FAIL'),
    (r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}', 'e-mail address', 'WARN'),
]
# where a name / e-mail is expected
PERSONAL_OK_FILES = {'LICENSE', 'CITATION.cff', 'README.md'}
# the scan's own pattern lists would match themselves
SELF = {os.path.join('tools', 'check_release.py')}

results = []


def report(status, check, msg):
    results.append((status, check, msg))


def walk(root):
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        for f in files:
            full = os.path.join(d, f)
            yield full, os.path.relpath(full, root)


def read_text(full):
    try:
        with open(full, encoding='utf-8') as f:
            return f.read()
    except (UnicodeDecodeError, OSError):
        return None


def notebook_text(nb, with_outputs=True):
    parts = []
    for c in nb.get('cells', []):
        s = c.get('source', '')
        parts.append(''.join(s) if isinstance(s, list) else s)
        if with_outputs:
            for o in c.get('outputs', []) or []:
                t = o.get('text')
                if t:
                    parts.append(''.join(t) if isinstance(t, list) else t)
                for v in (o.get('data') or {}).values():
                    if isinstance(v, (list, str)):
                        parts.append(''.join(v) if isinstance(v, list) else v)
    return '\n'.join(parts)


def check_files(root):
    bad = 0
    for full, rel in walk(root):
        name = os.path.basename(rel)
        posix = rel.replace(os.sep, '/')
        if any(fnmatch.fnmatch(name, p) for p in FORBIDDEN):
            report('FAIL', 'files', f'{posix}: must not be published (text / cache / weights)')
            bad += 1
        elif any(f'/{d}/' in f'/{posix}' for d in FORBIDDEN_DIRS):
            report('FAIL', 'files', f'{posix}: inside a run/checkpoint folder')
            bad += 1
        size = os.path.getsize(full)
        if size > 1_000_000:
            report('WARN', 'size', f'{posix}: {size / 1e6:.2f} MB, check what is in it')
    if not bad:
        report('PASS', 'files', 'no IndoSum text files, caches, checkpoints or weights')
    if not os.path.exists(os.path.join(root, '.gitignore')):
        report('WARN', 'files', '.gitignore missing')


def check_notebooks(root):
    n, bad = 0, 0
    for full, rel in walk(root):
        if not rel.endswith('.ipynb'):
            continue
        n += 1
        with open(full, encoding='utf-8') as f:
            nb = json.load(f)
        problems = []
        meta = nb.get('metadata', {})
        problems += [f'metadata.{k}' for k in ('widgets', 'colab', 'accelerator') if k in meta]
        n_out = n_exec = n_cmeta = 0
        for c in nb.get('cells', []):
            if c.get('outputs'):
                n_out += 1
            if c.get('execution_count') is not None:
                n_exec += 1
            if any(k in (c.get('metadata') or {}) for k in ('colab', 'outputId', 'executionInfo')):
                n_cmeta += 1
        if n_out:
            problems.append(f'{n_out} cells with outputs')
        if n_exec:
            problems.append(f'{n_exec} execution counts')
        if n_cmeta:
            problems.append(f'{n_cmeta} cells with Colab cell metadata')
        if problems:
            bad += 1
            report('FAIL', 'outputs', f'{rel}: ' + ', '.join(problems))
    if n and not bad:
        report('PASS', 'outputs', f'{n} notebooks stripped (no outputs, counts or Colab/widget metadata)')


def scan_patterns(root):
    found = {'secrets': 0, 'links': 0, 'personal': 0}
    for full, rel in walk(root):
        if rel in SELF or os.path.splitext(rel)[1].lower() not in TEXT_EXT:
            continue
        text = read_text(full)
        if text is None:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for pat, label in SECRETS:
                if re.search(pat, line):
                    report('FAIL', 'secrets', f'{rel}:{i} {label}')
                    found['secrets'] += 1
            for pat, label in LINKS:
                m = re.search(pat, line)
                if m:
                    report('FAIL', 'links', f'{rel}:{i} {label}: {m.group(0)[:80]}')
                    found['links'] += 1
            for pat, label, level in PERSONAL:
                m = re.search(pat, line)
                if not m:
                    continue
                if level == 'WARN' and os.path.basename(rel) in PERSONAL_OK_FILES:
                    continue
                report(level, 'personal', f'{rel}:{i} {label}: {m.group(0)[:80]}')
                found['personal'] += 1
    for k, label in [('secrets', 'no keys or tokens'), ('links', 'no private Colab/Drive links'),
                     ('personal', 'no local paths or e-mail addresses')]:
        if not found[k]:
            report('PASS', k, label)


def writefile_block(nb_path, target='mowoa_mo_shared.py'):
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    for c in nb.get('cells', []):
        if c.get('cell_type') != 'code':
            continue
        s = c.get('source', '')
        s = ''.join(s) if isinstance(s, list) else s
        m = re.match(r'%%writefile\s+(\S+)\s*\n', s)
        if m and os.path.basename(m.group(1)) == target:
            return s[m.end():]
    return None


def norm(text):
    return '\n'.join(l.rstrip() for l in text.replace('\r\n', '\n').strip().split('\n'))


def check_module(root):
    src = os.path.join(root, 'src', 'mowoa_mo_shared.py')
    if not os.path.exists(src):
        report('WARN', 'module', 'src/mowoa_mo_shared.py not found')
        return
    ref = norm(read_text(src))
    seen = 0
    for full, rel in walk(root):
        if not rel.endswith('.ipynb'):
            continue
        block = writefile_block(full)
        if block is None:
            continue
        seen += 1
        if norm(block) == ref:
            report('PASS', 'module', f'{rel}: %%writefile block == src/mowoa_mo_shared.py')
        else:
            report('FAIL', 'module', f'{rel}: %%writefile block differs from src/mowoa_mo_shared.py')
    if not seen:
        report('WARN', 'module', 'no notebook with %%writefile mowoa_mo_shared.py found')


WORD = re.compile(r'\w+', re.UNICODE)


def words(text):
    return [w.lower() for w in WORD.findall(text)]


def check_indosum(root, jsonl, n_articles, k):
    grams = set()
    with open(jsonl, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_articles:
                break
            row = json.loads(line)
            for seq in [s for p in row['paragraphs'] for s in p] + list(row['summary']):
                w = words(' '.join(seq))
                for j in range(len(w) - k + 1):
                    g = tuple(w[j:j + k])
                    if not all(x.isdigit() for x in g):
                        grams.add(g)
    hits = 0
    for full, rel in walk(root):
        ext = os.path.splitext(rel)[1].lower()
        if ext not in TEXT_EXT:
            continue
        if ext == '.ipynb':
            with open(full, encoding='utf-8') as f:
                text = notebook_text(json.load(f))
        else:
            text = read_text(full)
        if not text:
            continue
        w = words(text)
        for j in range(len(w) - k + 1):
            if tuple(w[j:j + k]) in grams:
                report('FAIL', 'indosum', f'{rel}: contains IndoSum text: "{" ".join(w[j:j + k])}"')
                hits += 1
                break
    if not hits:
        report('PASS', 'indosum', f'no {k}-token fragment of the first {n_articles} articles/references '
                                  f'({len(grams):,} fragments checked); images were not scanned')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', nargs='?', default='.')
    ap.add_argument('--indosum', help='path to test.01.jsonl, enables the text-leak scan')
    ap.add_argument('--n', type=int, default=150, help='number of evaluated articles (default 150)')
    ap.add_argument('--ngram', type=int, default=8, help='fragment length in tokens (default 8)')
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    check_files(root)
    check_notebooks(root)
    scan_patterns(root)
    check_module(root)
    if args.indosum:
        check_indosum(root, args.indosum, args.n, args.ngram)
    else:
        report('WARN', 'indosum', 'text-leak scan skipped (pass --indosum <test.01.jsonl>)')

    order = {'FAIL': 0, 'WARN': 1, 'PASS': 2}
    for status, check, msg in sorted(results, key=lambda r: (order[r[0]], r[1])):
        print(f'{status:<5} {check:<9} {msg}')
    n_fail = sum(1 for r in results if r[0] == 'FAIL')
    n_warn = sum(1 for r in results if r[0] == 'WARN')
    print(f'\n{n_fail} FAIL, {n_warn} WARN')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
