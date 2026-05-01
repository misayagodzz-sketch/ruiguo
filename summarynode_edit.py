"""
inject_summary.py

在 devign_parsed 的基础上，把 SummaryNode 注入到 nodes.csv 和 edges.csv，
输出到 devign_parsed_enhanced/，供 cpg_original.py 后续处理。

SummaryNode 注入规则：
  - 找 Callee 节点，查 callee_summaries.jsonl
  - 找 CFG 锚点（向上遍历 IS_AST_PARENT 找 isCFGNode=True 的祖先）
  - 在 nodes.csv 末尾追加 SummaryNode 行（isCFGNode=True）
  - 在 edges.csv 末尾追加 HAS_SUMMARY_* 边

SummaryNode 的 code 字段按 anchor 类型选择：
  EXPR: behavior + input + output
  COND: output only
  DECL/RET: output + behavior

用法:
    python inject_summary.py \
        --parsed_dir  /path/to/devign_parsed \
        --summary     /path/to/callee_summaries.jsonl \
        --output_dir  /path/to/devign_parsed_enhanced \
        --limit       10
"""

import os, re, json, argparse, logging, shutil
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

ANCHOR_NORMALIZE = {
    'ExpressionStatement':           'ExpressionStatement',
    'Statement':                     'ExpressionStatement',
    'PostIncDecOperationExpression': 'ExpressionStatement',
    'UnaryExpression':               'ExpressionStatement',
    'Condition':                     'Condition',
    'IdentifierDeclStatement':       'IdentifierDeclStatement',
    'AssignmentExpression':          'IdentifierDeclStatement',
    'ForInit':                       'IdentifierDeclStatement',
    'ReturnStatement':               'ReturnStatement',
}

ANCHOR_EDGE = {
    'ExpressionStatement':    'HAS_SUMMARY_EXPR',
    'Condition':              'HAS_SUMMARY_COND',
    'IdentifierDeclStatement':'HAS_SUMMARY_DECL',
    'ReturnStatement':        'HAS_SUMMARY_RET',
}


def select_text(s, anchor_type):
    behavior = s.get('behavior', '').strip()
    inp      = s.get('input',    '').strip()
    out      = s.get('output',   '').strip()
    if anchor_type == 'ExpressionStatement':
        parts = []
        if behavior: parts.append(behavior)
        if inp:      parts.append(f"Input: {inp}")
        if out:      parts.append(f"Output: {out}")
        return '. '.join(parts)
    elif anchor_type == 'Condition':
        return out if out else behavior
    else:
        parts = []
        if out:      parts.append(out)
        if behavior: parts.append(behavior)
        return '. '.join(parts)


def load_summary_db(path):
    db = {}
    with open(path) as f:
        for line in f:
            try:
                item = json.loads(line)
                fn = item.get('func_name', '').strip()
                if fn: db[fn] = item
            except Exception:
                pass
    logger.info(f"Summaries: {len(db)}")
    return db


def build_parent_map(ed):
    pm = {}
    for _, r in ed[ed['type'] == 'IS_AST_PARENT'].iterrows():
        pm[str(r.iloc[1]).strip()] = str(r.iloc[0]).strip()
    return pm


def cfg_ancestor(key, nodes_dict, pm, hops=8):
    cur = key
    for _ in range(hops):
        p = pm.get(cur)
        if not p: return None, None
        info = nodes_dict.get(p)
        if not info: return None, None
        if info.get('isCFGNode', '').strip() == 'True':
            return p, info.get('type', '').strip()
        cur = p
    return None, None


def process_one(src_dir, dst_dir, summary_db):
    """
    读取 src_dir/nodes.csv + edges.csv，
    注入 SummaryNode，写到 dst_dir/nodes.csv + edges.csv。
    返回注入数量。
    """
    nodes_src = os.path.join(src_dir, 'nodes.csv')
    edges_src = os.path.join(src_dir, 'edges.csv')
    if not os.path.exists(nodes_src) or not os.path.exists(edges_src):
        return 0

    nd = pd.read_csv(nodes_src, sep='\t', dtype=str, na_filter=False)
    ed = pd.read_csv(edges_src, sep='\t', dtype=str, na_filter=False)
    nd.columns = [c.strip() for c in nd.columns]
    ed.columns = [c.strip() for c in ed.columns]

    # 构建节点字典
    nodes_dict = {}
    for _, r in nd.iterrows():
        k = str(r.get('key', '')).strip()
        if k:
            nodes_dict[k] = {
                'type':      str(r.get('type', '')).strip(),
                'code':      str(r.get('code', '')).strip(),
                'isCFGNode': str(r.get('isCFGNode', '')).strip(),
            }

    pm = build_parent_map(ed)

    # 找注入点
    new_nodes = []   # 新增的SummaryNode行
    new_edges = []   # 新增的HAS_SUMMARY边行

    # 用已有key的最大数值作为新key的起点
    existing_keys = []
    for k in nodes_dict:
        try: existing_keys.append(int(k))
        except: pass
    next_key = max(existing_keys) + 1 if existing_keys else 9000000

    # nodes.csv的列顺序
    nd_cols = list(nd.columns)
    # edges.csv的列顺序
    ed_cols = list(ed.columns)

    seen_anchors = set()
    for _, r in nd.iterrows():
        if str(r.get('type', '')).strip() != 'Callee':
            continue
        fn = str(r.get('code', '')).strip()
        s  = summary_db.get(fn)
        if not s:
            continue
        k = str(r.get('key', '')).strip()
        ak, at_raw = cfg_ancestor(k, nodes_dict, pm)
        if ak is None:
            continue
        at = ANCHOR_NORMALIZE.get(at_raw)
        if at is None:
            continue

        # 避免同一锚点重复注入
        if ak in seen_anchors:
            continue
        seen_anchors.add(ak)

        text  = select_text(s, at)
        score = float(s.get('security_score', 0.0))
        etype = ANCHOR_EDGE[at]

        # 生成新节点key
        skey = str(next_key)
        next_key += 1

        # 构建SummaryNode行（用nodes.csv的列结构）
        node_row = {col: '' for col in nd_cols}
        node_row['key']       = skey
        node_row['type']      = 'SummaryNode'
        node_row['code']      = text[:500]   # 截断避免过长
        node_row['isCFGNode'] = 'True'       # ← 关键：让cpg_original保留它
        # 其余字段留空即可
        new_nodes.append(node_row)

        # 构建HAS_SUMMARY边（双向）
        # edges.csv列：start(col0), end(col1), type(col2), ...
        for src_k, dst_k in [(skey, ak), (ak, skey)]:
            edge_row = {col: '' for col in ed_cols}
            edge_row[ed_cols[0]] = src_k   # start
            edge_row[ed_cols[1]] = dst_k   # end
            edge_row[ed_cols[2]] = etype   # type
            new_edges.append(edge_row)

    # 写输出
    os.makedirs(dst_dir, exist_ok=True)

    if new_nodes:
        new_nd = pd.concat([nd, pd.DataFrame(new_nodes)], ignore_index=True)
    else:
        new_nd = nd
    if new_edges:
        new_ed = pd.concat([ed, pd.DataFrame(new_edges)], ignore_index=True)
    else:
        new_ed = ed

    new_nd.to_csv(os.path.join(dst_dir, 'nodes.csv'), sep='\t', index=False)
    new_ed.to_csv(os.path.join(dst_dir, 'edges.csv'), sep='\t', index=False)

    return len(new_nodes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parsed_dir',  required=True,
                   help='devign_parsed/ 原始目录')
    p.add_argument('--summary',     required=True,
                   help='callee_summaries.jsonl 路径')
    p.add_argument('--output_dir',  required=True,
                   help='输出目录 devign_parsed_enhanced/')
    p.add_argument('--limit',       type=int, default=None,
                   help='只处理前N个样本（调试）')
    args = p.parse_args()

    summary_db = load_summary_db(args.summary)
    os.makedirs(args.output_dir, exist_ok=True)

    names = sorted(os.listdir(args.parsed_dir))
    if args.limit:
        names = names[:args.limit]

    total_injected = 0
    skipped = 0

    for name in tqdm(names, desc='Injecting'):
        src = os.path.join(args.parsed_dir, name)
        dst = os.path.join(args.output_dir, name)
        if not os.path.isdir(src):
            continue
        try:
            n = process_one(src, dst, summary_db)
            total_injected += n
        except Exception as e:
            logger.debug(f"Failed {name}: {e}")
            skipped += 1

    logger.info(f"Done. total_injected={total_injected}, skipped={skipped}")
    logger.info(f"Output: {args.output_dir}")


if __name__ == '__main__':
    main()