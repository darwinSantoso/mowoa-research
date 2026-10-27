"""Check that a released notebook runs the same code as the original one.

    python tools/same_code_cells.py original/<name>.ipynb notebooks/<new>.ipynb

Code cells are compared in order on their Python syntax tree, so comments,
blank lines and formatting don't matter. Docstrings and print(...) statements
are also left out, since they only change what gets printed. Shell/magic lines
(!pip, %%writefile, ...) must match exactly. Markdown cells and outputs are
ignored.

Verdict:
  IDENTICAL                 same code
  IDENTICAL (TEXT LITERALS) same code apart from string literals, listed for review
  DIFFERENT                 code differs (exit code 1)
"""
import argparse
import ast
import difflib
import json
import sys

# cell magics whose body is plain Python
PY_CELL_MAGICS = ('%%writefile', '%%capture', '%%time', '%%timeit')


def code_cells(path):
    with open(path, encoding='utf-8') as f:
        nb = json.load(f)
    out = []
    for i, c in enumerate(nb['cells']):
        if c.get('cell_type') != 'code':
            continue
        src = c.get('source', '')
        src = ''.join(src) if isinstance(src, list) else src
        if src.strip():
            out.append((i, src))
    return out


def leading_indent(lines):
    # same as IPython: if the first line is indented, drop that prefix
    first = next((l for l in lines if l.strip()), '')
    prefix = first[:len(first) - len(first.lstrip())]
    if not prefix:
        return lines
    return [l[len(prefix):] if l.startswith(prefix) else l for l in lines]


def statement_starts(lines):
    """True for lines that begin a new statement: outside brackets and strings, and not after
    a backslash continuation. Only such lines can be IPython magics (a line inside brackets
    that starts with '!=' or '%' is an operator, not a shell command)."""
    starts, depth, quote, cont = [], 0, None, False
    for line in lines:
        starts.append(depth == 0 and quote is None and not cont)
        i, n = 0, len(line)
        while i < n:
            c = line[i]
            if quote:
                if c == '\\':
                    i += 2
                    continue
                if line.startswith(quote, i):
                    i += len(quote)
                    quote = None
                    continue
            elif c == '#':
                break
            elif line.startswith(('"""', "'''"), i):
                quote = line[i:i + 3]
                i += 3
                continue
            elif c in '"\'':
                quote = c
            elif c in '([{':
                depth += 1
            elif c in ')]}':
                depth = max(0, depth - 1)
            i += 1
        if quote in ('"', "'"):     # a single-quoted string can't span lines
            quote = None
        cont = line.rstrip().endswith('\\') and quote is None
    return starts


def split_magics(src):
    """-> (python source, list of magic/shell lines)"""
    lines = src.replace('\r\n', '\n').split('\n')
    magics = []
    if lines and lines[0].lstrip().startswith('%%'):
        magics.append(lines[0].strip())
        if not lines[0].lstrip().startswith(PY_CELL_MAGICS):
            # not Python inside (e.g. %%bash): compare the raw text
            magics.extend(l.rstrip() for l in lines[1:])
            return '', magics
        lines = lines[1:]
    lines = leading_indent(lines)
    py = []
    for l, first in zip(lines, statement_starts(lines)):
        s = l.lstrip()
        if first and s.startswith(('!', '%')):
            magics.append(s.rstrip())
            py.append(l[:len(l) - len(s)] + 'pass')   # keep the block structure
        else:
            py.append(l)
    return '\n'.join(py), magics


class Normalize(ast.NodeTransformer):
    """Drops docstrings and print(...) statements; optionally blanks all strings."""

    def __init__(self, blank_strings=False):
        self.blank_strings = blank_strings

    def visit_Expr(self, node):
        v = node.value
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == 'print':
            return None
        return self.generic_visit(node)

    def generic_visit(self, node):
        # docstring = first statement of a module/class/function body being a string
        body = getattr(node, 'body', None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:]
        super().generic_visit(node)
        if isinstance(getattr(node, 'body', None), list) and not node.body \
                and not isinstance(node, ast.Module):
            node.body = [ast.Pass()]
        return node

    def visit_Constant(self, node):
        if self.blank_strings and isinstance(node.value, str):
            return ast.Constant(value='<str>')
        return node

    def visit_JoinedStr(self, node):
        if self.blank_strings:
            return ast.Constant(value='<str>')
        return self.generic_visit(node)


def strings_of(tree):
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.append(repr(n.value))
        elif isinstance(n, ast.JoinedStr):
            try:
                out.append('f' + repr(ast.unparse(n)))
            except AttributeError:      # python < 3.9
                out.append('<f-string>')
    return out


def tree_of(py, blank_strings=False):
    tree = ast.parse(py)
    tree = Normalize(blank_strings).visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def compare(a_src, b_src):
    """-> ('same' | 'text' | 'diff', detail lines)"""
    a_py, a_mag = split_magics(a_src)
    b_py, b_mag = split_magics(b_src)
    if a_mag != b_mag:
        return 'diff', ['magic/shell lines differ:'] + [
            '  ' + l for l in difflib.unified_diff(a_mag, b_mag, lineterm='', n=0)][2:]
    try:
        ta, tb = tree_of(a_py), tree_of(b_py)
    except SyntaxError as e:
        same_text = a_py == b_py
        return ('same' if same_text else 'diff'), [f'could not parse ({e.msg}, line {e.lineno}); '
                                                  f'compared raw text: {"equal" if same_text else "different"}']
    if ast.dump(ta) == ast.dump(tb):
        return 'same', []
    if ast.dump(tree_of(a_py, True)) == ast.dump(tree_of(b_py, True)):
        sa, sb = strings_of(ta), strings_of(tb)
        lines = [l for l in difflib.unified_diff(sa, sb, lineterm='', n=0)
                 if not l.startswith(('---', '+++', '@@'))]
        return 'text', lines
    da = ast.dump(ta, indent=1).splitlines() if sys.version_info >= (3, 9) else ast.dump(ta).split(', ')
    db = ast.dump(tb, indent=1).splitlines() if sys.version_info >= (3, 9) else ast.dump(tb).split(', ')
    diff = [l for l in difflib.unified_diff(da, db, lineterm='', n=1)][2:40]
    return 'diff', diff


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('original')
    ap.add_argument('released')
    ap.add_argument('-q', '--quiet', action='store_true', help='only print the verdict')
    args = ap.parse_args()

    A, B = code_cells(args.original), code_cells(args.released)
    if len(A) != len(B):
        print(f'DIFFERENT: {len(A)} non-empty code cells in {args.original}, {len(B)} in {args.released}')
        return 1

    worst, report = 'same', []
    for (ia, sa), (ib, sb) in zip(A, B):
        kind, detail = compare(sa, sb)
        if kind != 'same':
            report.append((ia, ib, kind, detail))
        if kind == 'diff' or (kind == 'text' and worst == 'same'):
            worst = kind

    if not args.quiet:
        for ia, ib, kind, detail in report:
            tag = 'TEXT LITERALS' if kind == 'text' else 'CODE DIFFERS'
            print(f'cell {ia} -> cell {ib}: {tag}')
            for l in detail:
                print('    ' + l)
    verdict = {'same': 'IDENTICAL',
               'text': 'IDENTICAL (TEXT LITERALS) - review the strings above',
               'diff': 'DIFFERENT'}[worst]
    print(f'{verdict}   ({len(A)} code cells compared; comments, docstrings, print statements, '
          f'markdown and outputs ignored)')
    return 1 if worst == 'diff' else 0


if __name__ == '__main__':
    sys.exit(main())
