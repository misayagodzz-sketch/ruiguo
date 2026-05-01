"""
code_retrieval_vulnsc.py

从 Devign 原始数据出发，用 bare repo + git worktree 提取 callee 源码。
输出格式：每行一个 JSON，包含 idx, target, project, commit_id, func_name, callee 列表。

用法：
    python code_retrieval_vulnsc.py \
        --input  /path/to/devign.json \
        --repo_dir /path/to/repo \
        --output /path/to/output.jsonl \
        --n_layer 1 \
        --workers 4
"""

import os
import re
import json
import shutil
import tempfile
import argparse
import subprocess
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


REPO_MAP = {
    'FFmpeg': 'FFmpeg.git',
    'qemu':   'qemu.git',
}


# ── git worktree helpers ──────────────────────────────────────────────────────

def checkout_worktree(repo_dir, project, commit_id, work_dir):
    """
    在 work_dir 创建 worktree，checkout 到指定 commit。
    返回 work_dir（成功）或 None（失败）。
    """
    git_dir = os.path.join(repo_dir, REPO_MAP[project])
    result = subprocess.run(
        ['git', f'--git-dir={git_dir}', 'worktree', 'add',
         '--detach', work_dir, commit_id],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        return None
    return work_dir


def remove_worktree(repo_dir, project, work_dir):
    git_dir = os.path.join(repo_dir, REPO_MAP[project])
    subprocess.run(
        ['git', f'--git-dir={git_dir}', 'worktree', 'remove',
         '--force', work_dir],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)


# ── cscope helpers ────────────────────────────────────────────────────────────

def build_cscope_db(work_dir):
    """在 work_dir 里建 cscope 数据库，返回是否成功。"""
    c_files = subprocess.run(
        ['find', '.', '-name', '*.c'],
        stdout=subprocess.PIPE, text=True, cwd=work_dir
    ).stdout
    if not c_files.strip():
        return False
    with open(os.path.join(work_dir, 'cscope.files'), 'w') as f:
        f.write(c_files)
    r = subprocess.run(
        ['cscope', '-b', '-q', '-k', '-i', 'cscope.files'],
        cwd=work_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return r.returncode == 0


def get_callees_cscope(work_dir, func_name):
    """
    用 cscope 查询 func_name 调用了哪些函数（-L2）。
    返回 (file_path, [callee_names]) 或 ("", [])。
    """
    result = subprocess.run(
        ['cscope', '-dL2' + func_name],
        stdout=subprocess.PIPE, text=True, cwd=work_dir
    )
    lines = result.stdout.splitlines()
    if not lines:
        return "", []
    file_path = lines[0].split()[0]
    callees = []
    for line in lines:
        parts = line.split()
        if len(parts) > 1 and parts[1] not in callees:
            callees.append(parts[1])
    return file_path, callees


# ── ctags helpers ─────────────────────────────────────────────────────────────

def extract_func_name_from_file(file_path):
    """用 ctags 从单文件提取第一个函数名。"""
    result = subprocess.run(
        ['ctags', '--fields=+n', '-o', '-', '--sort=no',
         '--excmd=number', str(file_path)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    )
    lines = result.stdout.splitlines()
    if not lines:
        return ""
    return lines[0].split()[0]


def extract_func_source(work_dir, file_path, func_name):
    """用 ctags 定位 func_name 在 file_path 里的行号，返回源码字符串。"""
    full_path = os.path.join(work_dir, file_path.lstrip('./'))
    result = subprocess.run(
        ['ctags', '--fields=+n', '-o', '-', '--sort=no',
         '--excmd=number', full_path],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    )
    # 收集所有函数的起始行，用于推断 end 行
    func_lines = []  # list of (func_name, start_line)
    pattern_line = re.compile(r'line:(\d+)')
    for line in result.stdout.splitlines():
        parts = line.split('\t')
        if len(parts) < 4:
            continue
        name = parts[0]
        m = pattern_line.search(line)
        if m:
            func_lines.append((name, int(m.group(1))))

    # 找目标函数的起始行和下一个函数的起始行（作为end）
    for i, (name, start) in enumerate(func_lines):
        if name == func_name:
            end = func_lines[i+1][1] - 1 if i+1 < len(func_lines) else None
            try:
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    all_lines = f.readlines()
                return ''.join(all_lines[start-1:end])
            except Exception:
                return None
    return None


# ── main retrieval logic ──────────────────────────────────────────────────────

def retrieve_callees(work_dir, func_name, n_layer=1):
    """
    BFS 遍历调用树，收集最多 n_layer 层的 callee 源码。
    返回 list of dict: {layer, func_name, func_str, caller}
    """
    if not build_cscope_db(work_dir):
        return []

    callees = []
    visited = set()
    queue = [{'layer': 1, 'func_name': func_name, 'caller': func_name}]

    while queue:
        current = queue.pop(0)
        layer = current['layer']
        fname = current['func_name']

        if fname in visited:
            continue
        visited.add(fname)

        file_path, children = get_callees_cscope(work_dir, fname)
        if not file_path:
            continue

        func_str = extract_func_source(work_dir, file_path, fname)
        if func_str is None:
            continue

        if layer > 0:  # layer==0 是 root，不加入 callees
            callees.append({
                'layer': layer,
                'func_name': fname,
                'func_str': func_str,
                'caller': current['caller'],
            })

        if layer < n_layer:
            for child in children:
                if child not in visited:
                    queue.append({
                        'layer': layer + 1,
                        'func_name': child,
                        'caller': fname,
                    })

    return callees


def process_one(args):
    """处理单条数据，供多进程调用。"""
    idx, item, repo_dir, n_layer = args
    project = item['project']
    commit_id = item['commit_id']

    if project not in REPO_MAP:
        return None

    # 写 caller 函数到临时文件，提取函数名
    with tempfile.NamedTemporaryFile(suffix='.c', mode='w',
                                     delete=False, encoding='utf-8') as tf:
        tf.write(item['func'])
        tmp_path = tf.name
    func_name = extract_func_name_from_file(tmp_path)
    os.unlink(tmp_path)
    if not func_name:
        return None

    # checkout worktree
    work_dir = tempfile.mkdtemp(prefix=f'wr_{idx}_')
    try:
        if checkout_worktree(repo_dir, project, commit_id, work_dir) is None:
            return None
        callees = retrieve_callees(work_dir, func_name, n_layer)
        if not callees:
            return None
        return {
            'idx': idx,
            'target': item['target'],
            'project': project,
            'commit_id': commit_id,
            'func_name': func_name,
            'func': item['func'],
            'callee': callees,
        }
    finally:
        remove_worktree(repo_dir, project, work_dir)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',    required=True,  help='devign.json path')
    parser.add_argument('--repo_dir', required=True,  help='dir containing FFmpeg.git and qemu.git')
    parser.add_argument('--output',   required=True,  help='output jsonl path')
    parser.add_argument('--n_layer',  type=int, default=1, help='callee depth (1 recommended)')
    parser.add_argument('--workers',  type=int, default=4, help='parallel workers')
    parser.add_argument('--limit',    type=int, default=None, help='debug: only process N items')
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    if args.limit:
        data = data[:args.limit]

    tasks = [(i, item, args.repo_dir, args.n_layer) for i, item in enumerate(data)]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    success = 0
    with open(args.output, 'w') as out_f:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one, t): t[0] for t in tasks}
            for future in tqdm(as_completed(futures), total=len(tasks)):
                result = future.result()
                if result is not None:
                    out_f.write(json.dumps(result) + '\n')
                    success += 1

    print(f'Done. {success}/{len(tasks)} samples have callees.')
    print(f'Output: {args.output}')


if __name__ == '__main__':
    main()