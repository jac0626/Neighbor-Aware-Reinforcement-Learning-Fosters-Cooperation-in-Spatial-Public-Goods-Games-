"""Place source credits below thesis floats without changing data or artwork.

Run from any directory after exporting new tables. Raw experiment outputs and
exporter snapshots are left intact; the typesetting transform is recorded.
"""
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MARKER = r'\par\smallskip{\zihao{-5}\rmfamily 来源：'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


changes = {}
floats = 0
for folder in ['chapters', 'backmatter', 'figures']:
    for path in sorted((ROOT / folder).rglob('*.tex')):
        original = path.read_text()

        def format_float(match):
            global floats
            block = match.group(0)
            floats += 1
            if MARKER in block:
                return block
            if path.name == '03-work1.tex':
                credit = r'Kang等，文献\cite{kang2025neighbor}，2025年。'
                if 'fig:w1-neighborhood' in block:
                    credit += '作者依据模型定义绘制。'
                block = block.replace(r' 原图引自文献\cite{kang2025neighbor}。', '')
                block = block.replace(r'各快照引自文献\cite{kang2025neighbor}，按时间重新排版。', '各快照按原始时间节点重新排版。')
            elif path.name in ['04-work2.tex', 'appendix.tex']:
                credit = '作者，本文模型定义与实验配置，2026年。'
            elif 'work2-final' in path.parts or path.name in ['05-final-test.tex', 'final-supplement.tex']:
                credit = '作者，本文冻结后的留出测试，每个条件30次独立运行，2026年。'
            elif 'work2-timing' in path.parts:
                credit = '作者，本文单进程计时实验，2026年。'
            else:
                credit = '作者，本文模型开发与仿真实验，2026年。'
            block = block.replace('数据来源：本文冻结控制器后的泛化开发实验。', '')
            block = block.replace('数据来源：本文两批泛化开发实验，重复条件使用同一原始记录。', '重复条件使用同一原始记录。')
            block = block.replace('数据来源：本文泛化开发实验。', '')
            block = re.sub(r'\\par\\smallskip\\footnotesize 数据来源：[^\n]*\n', '', block)
            end = r'\end{' + match.group(1) + '}'
            return block.replace(end, MARKER + credit + r'\par}' + '\n' + end)

        formatted = re.sub(r'\\begin\{(figure|table)\}.*?\\end\{\1\}', format_float, original, flags=re.S)
        # Numeric rows must survive the source-credit edit byte for byte.
        assert [x for x in original.splitlines() if '&' in x] == [x for x in formatted.splitlines() if '&' in x]
        if formatted != original:
            before = digest(path)
            path.write_text(formatted)
            changes[str(path.relative_to(ROOT))] = {'before': before, 'after': digest(path)}

# Existing provenance keeps the raw-data hashes and identifies this additional
# presentation-only transformation separately from the statistical exporter.
for path in (ROOT / 'validation').glob('work2-*-sources.json'):
    record = json.loads(path.read_text())
    key = 'exported_files' if 'exported_files' in record else 'artifacts' if 'artifacts' in record else None
    if key is None:
        continue
    tag = path.name.removesuffix('-sources.json')
    directory = ROOT / 'figures' / tag
    changed = [name for name in record[key] if (directory / name).exists() and digest(directory / name) != record[key][name]]
    if changed:
        for name in changed:
            record.setdefault('before_source_typesetting', {}).setdefault(name, record[key][name])
        for name in changed:
            record[key][name] = digest(directory / name)
    if changed or 'before_source_typesetting' in record:
        record['source_typesetting_script_sha256'] = digest(Path(__file__))
        record['source_typesetting_note'] = 'Source credits moved below floats; data rows and artwork unchanged.'
        path.write_text(json.dumps(record, indent=2) + '\n')

log = ROOT / 'validation' / 'figure-source-formatting.json'
record = json.loads(log.read_text()) if log.exists() else {'files': {}}
record['files'].update(changes)
for name, hashes in record['files'].items():
    hashes['after'] = digest(ROOT / name)
record['script_sha256'] = digest(Path(__file__))
record['floats_checked'] = floats
log.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'floats_checked': floats, 'changed_files': len(changes)}))
