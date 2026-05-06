import os, re, json, shutil, tempfile, argparse, subprocess
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_MAP = {'FFmpeg': 'FFmpeg.git', 'qemu': 'qemu.git'}

def checkout_worktree(repo_dir, project, commit_id, work_dir):
    git_dir = os.path.join(repo_dir, REPO_MAP[project])
    r = subprocess.run(['git', f'--git-dir={git_dir}', 'worktree', 'add', '--detach', work_dir, commit_id],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    return work_dir if r.returncode == 0 else None

def remove_worktree(repo_dir, project, work_dir):
    git_dir = os.path.join(repo_dir, REPO_MAP[project])
    subprocess.run(['git', f'--git-dir={git_dir}', 'worktree', 'remove', '--force', work_dir],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(work_dir, ignore_errors=True)

def build_cscope_db(work_dir):
    c_files = subprocess.run(['find', '.', '-name', '*.c'], stdout=subprocess.PIPE, text=True, cwd=work_dir).stdout
    if not c_files.strip():
        return False
    with open(os.path.join(work_dir, 'cscope.files'), 'w') as f:
        f.write(c_files)
    r = subprocess.run(['cscope', '-b', '-q', '-k', '-i', 'cscope.files'],
        cwd=work_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0

def get_callees_cscope(work_dir, func_name):
    r = subprocess.run(['cscope', '-dL2' + func_name], stdout=subprocess.PIPE, text=True, cwd=work_dir)
    lines = r.stdout.splitlines()
    if not lines:
        return [], ""
    file_path = lines[0].split()[0]
    callees = []
    for line in lines:
        parts = line.split()
        if len(parts) > 1 and parts[1] not in callees:
            callees.append(parts[1])
    return callees, file_path

def find_func_definition(work_dir, func_name):
    r = subprocess.run(['cscope', '-dL1' + func_name], stdout=subprocess.PIPE, text=True, cwd=work_dir)
    lines = r.stdout.splitlines()
    return lines[0].split()[0] if lines else ""

def extract_func_name_from_file(file_path):
    r = subprocess.run(['ctags', '--fields=+n', '-o', '-', '--sort=no', '--excmd=number', str(file_path)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    lines = r.stdout.splitlines()
    return lines[0].split()[0] if lines else ""

def extract_func_source(work_dir, file_path, func_name):
    full_path = os.path.join(work_dir, file_path.lstrip('./'))
    r = subprocess.run(['ctags', '--fields=+n', '-o', '-', '--sort=no', '--excmd=number', full_path],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    pattern = re.compile(r'line:(\d+)')
    func_lines = []
    for line in r.stdout.splitlines():
        parts = line.split('\t')
        if len(parts) < 4:
            continue
        m = pattern.search(line)
        if m:
            func_lines.append((parts[0], int(m.group(1))))
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

def retrieve_callees(work_dir, func_name, n_layer=1):
    if not build_cscope_db(work_dir):
        return []
    callees = []
    visited = set([func_name])
    direct_callees, _ = get_callees_cscope(work_dir, func_name)
    queue = [{'layer': 1, 'func_name': c, 'caller': func_name} for c in direct_callees]
    while queue:
        current = queue.pop(0)
        layer = current['layer']
        fname = current['func_name']
        if fname in visited:
            continue
        visited.add(fname)
        def_path = find_func_definition(work_dir, fname)
        if not def_path:
            continue
        func_str = extract_func_source(work_dir, def_path, fname)
        if func_str is None:
            continue
        callees.append({'layer': layer, 'func_name': fname, 'func_str': func_str, 'caller': current['caller']})
        if layer < n_layer:
            children, _ = get_callees_cscope(work_dir, fname)
            for child in children:
                if child not in visited:
                    queue.append({'layer': layer+1, 'func_name': child, 'caller': fname})
    return callees

def process_one(args):
    idx, item, repo_dir, n_layer = args
    if item['project'] not in REPO_MAP:
        return None
    with tempfile.NamedTemporaryFile(suffix='.c', mode='w', delete=False, encoding='utf-8') as tf:
        tf.write(item['func'])
        tmp_path = tf.name
    func_name = extract_func_name_from_file(tmp_path)
    os.unlink(tmp_path)
    if not func_name:
        return None
    work_dir = tempfile.mkdtemp(prefix=f'wr_{idx}_')
    try:
        if checkout_worktree(repo_dir, item['project'], item['commit_id'], work_dir) is None:
            return None
        callees = retrieve_callees(work_dir, func_name, n_layer)
        if not callees:
            return None
        return {'idx': idx, 'target': item['target'], 'project': item['project'],
                'commit_id': item['commit_id'], 'func_name': func_name,
                'func': item['func'], 'callee': callees}
    finally:
        remove_worktree(repo_dir, item['project'], work_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--repo_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n_layer', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=None)
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

if __name__ == '__main__':
    main()
