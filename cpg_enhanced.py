"""
cpg_enhanced.py v4

按anchor类型选择性对齐summary字段，去掉effects，SummaryNode特征：
  CodeBERT(selected_text, 768d) + score(1d) + anchor_onehot(4d) = 773d

selected_text按anchor类型：
  EXPR: behavior + input + output  (全量)
  COND: output only               (只关心返回值)
  DECL: output + behavior         (拿到什么 + 性质)
  RET:  output + behavior         (转发什么语义)

普通节点特征：
  word2vec(code, 100d) + zeros(673d) = 773d

输出格式与AMPLE兼容：{"node_features":..., "graph":..., "targets":...}
"""

import os, re, json, argparse, logging
from typing import Optional
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────────────────

ANCHOR_TYPES = [
    "ExpressionStatement",
    "Condition",
    "IdentifierDeclStatement",
    "ReturnStatement",
]
ANCHOR_IDX = {t: i for i, t in enumerate(ANCHOR_TYPES)}

# Joern实际CFG节点类型 → 归并到4类anchor
# 不在此表的类型skip（Label/Parameter/Break/Goto等与callsite无关）
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

WV_DIM       = 100
CODEBERT_DIM = 768
SCORE_DIM    = 1
ANCHOR_DIM   = len(ANCHOR_TYPES)   # 4
FEAT_DIM     = CODEBERT_DIM + SCORE_DIM + ANCHOR_DIM  # 773

EDGE_TYPE_MAP = {
    'IS_AST_PARENT': 1, 'FLOWS_TO': 3, 'DEF': 4, 'USE': 5,
    'REACHES': 6, 'CONTROLS': 7, 'DOM': 9, 'POST_DOM': 10,
    'HAS_SUMMARY_EXPR': 13,
    'HAS_SUMMARY_COND': 14,
    'HAS_SUMMARY_DECL': 15,
    'HAS_SUMMARY_RET':  16,
    'HAS_SUMMARY_USE':  17,
}

# anchor类型 → 对应的HAS_SUMMARY边类型
ANCHOR_EDGE = {
    "ExpressionStatement":    'HAS_SUMMARY_EXPR',
    "Condition":              'HAS_SUMMARY_COND',
    "IdentifierDeclStatement":'HAS_SUMMARY_DECL',
    "ReturnStatement":        'HAS_SUMMARY_RET',
}

# ── 按anchor类型选择summary文本 ────────────────────────────────────────────────

def select_summary_text(summary: dict, anchor_type: str) -> str:
    behavior = summary.get('behavior', '').strip()
    inp      = summary.get('input', '').strip()
    out      = summary.get('output', '').strip()

    if anchor_type == 'ExpressionStatement':
        # 全量：副作用全部相关
        parts = []
        if behavior: parts.append(behavior)
        if inp:      parts.append(f"Input: {inp}")
        if out:      parts.append(f"Output: {out}")
        return '. '.join(parts)

    elif anchor_type == 'Condition':
        # 只关心返回值语义
        return out if out else behavior

    elif anchor_type == 'IdentifierDeclStatement':
        # 拿到什么 + 性质
        parts = []
        if out:      parts.append(out)
        if behavior: parts.append(behavior)
        return '. '.join(parts)

    elif anchor_type == 'ReturnStatement':
        # 转发什么语义给调用方
        parts = []
        if out:      parts.append(out)
        if behavior: parts.append(behavior)
        return '. '.join(parts)

    else:
        return behavior


# ── CodeBERT 编码器 ────────────────────────────────────────────────────────────

class CodeBERTEncoder:
    def __init__(self, model_name: str, device: str, batch_size: int = 32):
        logger.info(f"Loading CodeBERT: {model_name}")
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)
        self.model      = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device     = device
        self.batch_size = batch_size

    @torch.no_grad()
    def encode(self, texts: list) -> np.ndarray:
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors='pt'
            ).to(self.device)
            out = self.model(**inputs)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
            results.append(cls)
        return np.vstack(results)


# ── word2vec 编码（普通节点用）────────────────────────────────────────────────

def tokenize(code):
    toks = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+',
                      re.sub(r'[^a-zA-Z0-9_]', ' ', code))
    return toks if toks else ['UNK']

def wv_encode(wv, text):
    vecs = [wv.wv[t] for t in tokenize(text) if t in wv.wv]
    return np.mean(vecs, axis=0).astype(np.float32) if vecs \
           else np.zeros(WV_DIM, np.float32)


# ── 特征构建 ──────────────────────────────────────────────────────────────────

def normal_feat(wv, code: str) -> np.ndarray:
    """普通节点: [wv(100) | zeros(673)] = 773d"""
    v = wv_encode(wv, code)
    return np.concatenate([v, np.zeros(FEAT_DIM - WV_DIM, np.float32)])

def summary_feat(codebert_vec: np.ndarray, score: float,
                 anchor_type: str) -> np.ndarray:
    """SummaryNode: [CodeBERT(768) | score(1) | anchor_onehot(4)] = 773d"""
    sc  = np.array([score], np.float32)
    anc = np.zeros(ANCHOR_DIM, np.float32)
    idx = ANCHOR_IDX.get(anchor_type, -1)
    if idx >= 0:
        anc[idx] = 1.0
    return np.concatenate([codebert_vec, sc, anc])


# ── CPG 加载 ──────────────────────────────────────────────────────────────────

def load_cpg(d):
    nd = pd.read_csv(f'{d}/nodes.csv', sep='\t', dtype=str, na_filter=False)
    ed = pd.read_csv(f'{d}/edges.csv', sep='\t', dtype=str, na_filter=False)
    nd.columns = [c.strip() for c in nd.columns]
    ed.columns = [c.strip() for c in ed.columns]
    return nd, ed

def build_parent_map(edges_df):
    pm = {}
    for _, r in edges_df[edges_df['type'] == 'IS_AST_PARENT'].iterrows():
        pm[str(r.iloc[1]).strip()] = str(r.iloc[0]).strip()
    return pm

def cfg_ancestor(key, nodes_dict, pm, hops=8):
    cur = key
    for _ in range(hops):
        p = pm.get(cur)
        if not p: return None, None
        info = nodes_dict.get(p)
        if not info: return None, None
        if info['isCFGNode'] == 'True':
            return p, info['type']
        cur = p
    return None, None


# ── USE节点查找（DECL专用）───────────────────────────────────────────────────

def find_use_nodes(anchor_key, nodes_dict, edges_df):
    """
    DECL情形：沿 anchor -DEF-> Symbol -USE-> CFG节点 路径
    找到所有USE点（isCFGNode=True的下游节点）
    """
    use_nodes = []

    # 1. anchor -DEF-> Symbol
    def_edges = edges_df[
        (edges_df['type'] == 'DEF') &
        (edges_df.iloc[:, 0].str.strip() == anchor_key)
    ]
    symbol_keys = [str(r.iloc[1]).strip() for _, r in def_edges.iterrows()]

    # 2. Symbol -USE-> CFG节点
    for sk in symbol_keys:
        use_edges = edges_df[
            (edges_df['type'] == 'USE') &
            (edges_df.iloc[:, 0].str.strip() == sk)
        ]
        for _, r in use_edges.iterrows():
            dst = str(r.iloc[1]).strip()
            info = nodes_dict.get(dst)
            if info and info['isCFGNode'] == 'True' and dst != anchor_key:
                use_nodes.append(dst)

    return list(set(use_nodes))


# ── 单样本处理（收集文本，不在这里编码）──────────────────────────────────────

def collect_sample(name, parsed_dir, summary_db, max_nodes):
    d = os.path.join(parsed_dir, name)
    if not os.path.isdir(d):
        d += '.c'
    if not os.path.isdir(d):
        return None
    try:
        nd, ed = load_cpg(d)
    except Exception:
        return None
    if nd.empty:
        return None

    nodes_dict = {}
    for _, r in nd.iterrows():
        k = str(r.get('key', '')).strip()
        if k:
            nodes_dict[k] = {
                'type':      str(r.get('type', '')).strip(),
                'code':      str(r.get('code', '')).strip(),
                'isCFGNode': str(r.get('isCFGNode', '')).strip(),
            }

    if len(nodes_dict) > max_nodes:
        return None

    pm = build_parent_map(ed)

    # 找注入点
    injections = []
    for k, info in nodes_dict.items():
        if info['type'] != 'Callee':
            continue
        fn = info['code'].strip()
        s  = summary_db.get(fn)
        if not s:
            continue
        ak, at_raw = cfg_ancestor(k, nodes_dict, pm)
        if ak is None:
            continue

        # 归并到4类anchor，skip无关类型
        at = ANCHOR_NORMALIZE.get(at_raw)
        if at is None:
            continue

        # 按anchor类型选择文本
        sel_text = select_summary_text(s, at)

        # DECL额外找USE节点
        use_keys = []
        if at == 'IdentifierDeclStatement':
            use_keys = find_use_nodes(ak, nodes_dict, ed)

        injections.append({
            'func_name':   fn,
            'anchor_key':  ak,
            'anchor_type': at,
            'sel_text':    sel_text,
            'score':       float(s.get('security_score', 0.0)),
            'use_keys':    use_keys,
        })

    return {
        'name':       name,
        'nodes_dict': nodes_dict,
        'edges_df':   ed,
        'injections': injections,
    }


def build_graph(sample, wv, codebert_vecs):
    """
    拿到CodeBERT编码后，组装节点特征矩阵和边列表。
    codebert_vecs: list of np.ndarray，与 injections 一一对应
    """
    nodes_dict  = sample['nodes_dict']
    ed          = sample['edges_df']
    injections  = sample['injections']

    # 节点列表
    node_keys  = list(nodes_dict.keys())
    k2i        = {k: i for i, k in enumerate(node_keys)}
    node_feats = [normal_feat(wv, nodes_dict[k]['code']).tolist()
                  for k in node_keys]

    # 插入SummaryNode
    for inj, cb_vec in zip(injections, codebert_vecs):
        idx  = len(node_feats)
        skey = f'__sum_{idx}__'
        k2i[skey] = idx
        node_feats.append(
            summary_feat(cb_vec, inj['score'], inj['anchor_type']).tolist()
        )
        inj['sidx'] = idx
        inj['skey'] = skey

    # 原始边
    graph = []
    for _, r in ed.iterrows():
        et = str(r.get('type', '')).strip()
        if et not in EDGE_TYPE_MAP:
            continue
        si = k2i.get(str(r.iloc[0]).strip())
        di = k2i.get(str(r.iloc[1]).strip())
        if si is None or di is None:
            continue
        graph.append([si, di, EDGE_TYPE_MAP[et]])

    # HAS_SUMMARY边
    for inj in injections:
        ai  = k2i.get(inj['anchor_key'])
        si  = inj.get('sidx')
        if ai is None or si is None:
            continue
        et  = EDGE_TYPE_MAP[ANCHOR_EDGE.get(inj['anchor_type'], 'HAS_SUMMARY_EXPR')]
        graph.extend([[si, ai, et], [ai, si, et]])

        # DECL专属：额外连接USE节点
        if inj['anchor_type'] == 'IdentifierDeclStatement':
            for uk in inj['use_keys']:
                ui = k2i.get(uk)
                if ui is not None:
                    ue = EDGE_TYPE_MAP['HAS_SUMMARY_USE']
                    graph.extend([[si, ui, ue], [ui, si, ue]])

    return {
        'node_features': node_feats,
        'graph':         graph,
        'meta': {
            'name':      sample['name'],
            'n_nodes':   len(node_feats),
            'n_summary': len(injections),
        }
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

def load_summary_db(path):
    db = {}
    with open(path) as f:
        for line in f:
            try:
                item = json.loads(line)
                fn = item.get('func_name', '').strip()
                if fn:
                    db[fn] = item
            except Exception:
                pass
    logger.info(f"Summaries loaded: {len(db)}")
    return db


def build_split(entries, parsed_dir, summary_db, wv, encoder,
                output_path, max_nodes, limit):
    if limit:
        entries = entries[:limit]

    dataset   = []
    skipped   = 0
    no_inject = 0

    for entry in tqdm(entries, desc=os.path.basename(output_path)):
        name  = entry['file_path']
        label = int(entry['label'])

        sample = collect_sample(name, parsed_dir, summary_db, max_nodes)
        if sample is None:
            skipped += 1
            continue

        injections = sample['injections']
        if not injections:
            no_inject += 1
            # 没有注入点也保留，SummaryNode数量=0
            codebert_vecs = []
        else:
            texts = [inj['sel_text'] if inj['sel_text'] else 'unknown function'
                     for inj in injections]
            try:
                codebert_vecs = list(encoder.encode(texts))
            except Exception as e:
                logger.debug(f"CodeBERT failed {name}: {e}")
                skipped += 1
                continue

        result = build_graph(sample, wv, codebert_vecs)
        result['targets'] = [[label]]
        dataset.append(result)

    logger.info(
        f"{os.path.basename(output_path)}: "
        f"built={len(dataset)}, skipped={skipped}, no_injection={no_inject}"
    )
    with open(output_path, 'w') as f:
        json.dump(dataset, f)
    logger.info(f"Saved → {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parsed_dir',  required=True)
    p.add_argument('--summary',     required=True)
    p.add_argument('--split_dir',   required=True)
    p.add_argument('--wv_model',    required=True)
    p.add_argument('--output_dir',  required=True)
    p.add_argument('--codebert',    default='microsoft/codebert-base')
    p.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch_size',  type=int, default=32)
    p.add_argument('--max_nodes',   type=int, default=500)
    p.add_argument('--limit',       type=int, default=None)
    p.add_argument('--splits',      nargs='+', default=['train', 'valid', 'test'])
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    summary_db = load_summary_db(args.summary)

    logger.info(f"Loading Word2Vec: {args.wv_model}")
    wv = Word2Vec.load(args.wv_model)

    encoder = CodeBERTEncoder(args.codebert, args.device, args.batch_size)

    config = {
        'feat_dim':     FEAT_DIM,
        'codebert_dim': CODEBERT_DIM,
        'wv_dim':       WV_DIM,
        'score_dim':    SCORE_DIM,
        'anchor_dim':   ANCHOR_DIM,
        'anchor_types': ANCHOR_TYPES,
        'edge_types':   EDGE_TYPE_MAP,
        'anchor_text_selection': {
            'ExpressionStatement':    'behavior + input + output',
            'Condition':              'output only',
            'IdentifierDeclStatement':'output + behavior',
            'ReturnStatement':        'output + behavior',
        }
    }
    with open(f'{args.output_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)

    for split in args.splits:
        sf = f'{args.split_dir}/{split}_raw_code.json'
        if not os.path.exists(sf):
            logger.warning(f"Not found: {sf}")
            continue
        entries = json.load(open(sf))
        out     = f'{args.output_dir}/devign-{split}-v0.json'
        build_split(entries, args.parsed_dir, summary_db, wv, encoder,
                    out, args.max_nodes, args.limit)

    logger.info(f"Done. FEAT_DIM={FEAT_DIM}")
    logger.info(f"Anchor text selection: EXPR=full, COND=output_only, DECL/RET=output+behavior")

if __name__ == '__main__':
    main()