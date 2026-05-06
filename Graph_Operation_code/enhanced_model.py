#!/usr/bin/env python3
"""
AMPLE + 跨函数语义注入 模型改造

包含:
1. EffectEncoder - 把 effect 标签编码成可学习的向量
2. SummaryInjector - 在 GNN 输入前注入 effect 向量到目标节点
3. SAMP-enhanced GraphTransformerLayer - 双通道消息传递
4. 改造后的 DevignModel
5. 改造后的 DataEntry/DataSet - 加载 injection_plan
6. 改造后的 GGNNBatchGraph - 传递 injection_mask

使用方法:
  将此文件放到 AMPLE_code/ 目录下
  修改 main.py 中的 import 来使用这些改造后的类
"""

import copy
import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.nn.pytorch import RelGraphConv
from tqdm import tqdm

from utils import load_default_identifiers, initialize_batch, debug
from graph_transformer_layers import GraphTransformerLayer, Norm, MultiHeadAttentionLayer, src_dot_dst, scaled_exp
from mlp_readout import MLPReadout


# ============================================================
# 1. Effect 编码器
# ============================================================

# Effect 类型列表（和 behavior_decomposer.py 一致）
EFFECT_TYPES = ['FREE', 'ALLOC', 'WRITE', 'READ', 'NULL', 'CHECK', 'COPY', 'INIT', 'LOCK', 'UNLOCK']
EFFECT_TO_IDX = {e: i for i, e in enumerate(EFFECT_TYPES)}


class EffectEncoder(nn.Module):
    """
    把 effect 标签编码成可学习的向量。
    每种 effect 类型有一个可学习的 embedding。
    多个 effect 取加权和。
    """
    def __init__(self, effect_dim=100, n_effects=len(EFFECT_TYPES)):
        super().__init__()
        self.effect_embeddings = nn.Embedding(n_effects, effect_dim)
        nn.init.xavier_uniform_(self.effect_embeddings.weight)
    
    def forward(self, effect_indices):
        """
        effect_indices: list of lists，每个元素是一个节点的 effect 索引列表
        返回: [n_nodes, effect_dim] 的张量
        """
        device = self.effect_embeddings.weight.device
        result = torch.zeros(len(effect_indices), self.effect_embeddings.embedding_dim, device=device)
        
        for i, indices in enumerate(effect_indices):
            if indices:
                idx_tensor = torch.LongTensor(indices).to(device)
                embeddings = self.effect_embeddings(idx_tensor)
                result[i] = embeddings.mean(dim=0)  # 多个 effect 取平均
        
        return result


# ============================================================
# 2. Summary Injector (门控注入模块)
# ============================================================

class SummaryInjector(nn.Module):
    """
    把 effect 向量注入到目标节点的特征中。
    
    h_i' = h_i + gate(h_i, e_i) * W_s * e_i
    
    其中:
    - h_i: 原始节点特征
    - e_i: effect 编码向量
    - gate: 可学习的标量门控
    - W_s: 线性变换
    """
    def __init__(self, feature_dim=100, effect_dim=100):
        super().__init__()
        self.W_s = nn.Linear(effect_dim, feature_dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim + effect_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, node_features, effect_vectors, injection_mask):
        """
        node_features: [n_nodes, feature_dim]
        effect_vectors: [n_nodes, effect_dim] (非注入节点为零向量)
        injection_mask: [n_nodes] (0/1)
        
        返回: 注入后的 node_features [n_nodes, feature_dim]
        """
        # 只对被注入的节点计算
        mask = injection_mask.unsqueeze(1).float()  # [n_nodes, 1]
        
        # 门控
        concat = torch.cat([node_features, effect_vectors], dim=1)  # [n_nodes, feature_dim + effect_dim]
        g = self.gate(concat)  # [n_nodes, 1]
        
        # 注入
        transformed = self.W_s(effect_vectors)  # [n_nodes, feature_dim]
        injection = g * transformed * mask  # 只对 mask=1 的节点生效
        
        return node_features + injection


# ============================================================
# 3. SAMP-enhanced MultiHeadAttentionLayer
# ============================================================

class SAMPMultiHeadAttentionLayer(nn.Module):
    """
    Summary-Aware Message Passing 多头注意力层。
    
    对来自被注入节点的消息使用不同的权重矩阵：
    msg = (1 - m_j) * W1*h_j + m_j * W2*h_j
    """
    def __init__(self, in_dim, out_dim, num_heads, use_bias=True):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        # 普通通道 (和原始 AMPLE 一样)
        self.feature_Q = RelGraphConv(in_feat=in_dim, out_feat=num_heads*out_dim, num_rels=9, 
                                       regularizer='basis', num_bases=9, 
                                       activation=F.relu, dropout=0.1)
        self.feature_K = RelGraphConv(in_feat=in_dim, out_feat=num_heads*out_dim, num_rels=9, 
                                       regularizer='basis', num_bases=9, 
                                       activation=F.relu, dropout=0.1)
        self.feature_V = RelGraphConv(in_feat=in_dim, out_feat=num_heads*out_dim, num_rels=9, 
                                       regularizer='basis', num_bases=9, 
                                       activation=F.relu, dropout=0.1)
        
        # 摘要通道 (只有 V 用不同的权重，Q 和 K 共享)
        self.feature_V_summary = RelGraphConv(in_feat=in_dim, out_feat=num_heads*out_dim, num_rels=9, 
                                               regularizer='basis', num_bases=9, 
                                               activation=F.relu, dropout=0.1)
    
    def propagate_attention(self, g):
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
    
    def forward(self, g, h, e, injection_mask=None):
        """
        g: DGLGraph
        h: 节点特征 [n_nodes, in_dim]
        e: 边类型
        injection_mask: [n_nodes] 注入标记 (0/1)，None 时退化为普通注意力
        """
        feature_Q = self.feature_Q(g, h, e)
        feature_K = self.feature_K(g, h, e)
        
        if injection_mask is not None and injection_mask.sum() > 0:
            # 双通道: 根据 injection_mask 混合两种 V
            V_normal = self.feature_V(g, h, e)
            V_summary = self.feature_V_summary(g, h, e)
            
            mask = injection_mask.unsqueeze(1).float()  # [n_nodes, 1]
            feature_V = (1 - mask) * V_normal + mask * V_summary
        else:
            feature_V = self.feature_V(g, h, e)
        
        Q_h = feature_Q
        K_h = feature_K
        V_h = feature_V
        
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        
        self.propagate_attention(g)
        
        head_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))
        return head_out


# ============================================================
# 4. SAMP-enhanced GraphTransformerLayer
# ============================================================

class SAMPGraphTransformerLayer(nn.Module):
    """
    带 SAMP 的 Graph Transformer 层。
    和原始 GraphTransformerLayer 的区别：
    - 使用 SAMPMultiHeadAttentionLayer 替代 MultiHeadAttentionLayer
    - forward 额外接收 injection_mask 参数
    """
    def __init__(self, input_dim, output_dim, max_edge_types, num_heads, 
                 num_steps=8, dropout=0.0, layer_norm=False, batch_norm=True, 
                 residual=False, use_bias=True):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        
        # SAMP 注意力层
        self.attention = SAMPMultiHeadAttentionLayer(input_dim, output_dim // num_heads, num_heads, use_bias)
        
        self.O = nn.Linear(output_dim, output_dim)
        
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(output_dim)
            self.Graph_norm1 = Norm(hidden_dim=output_dim)
        
        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(output_dim)
        
        # FFN
        self.FFN_layer1 = nn.Linear(output_dim, output_dim * 2)
        self.FFN_layer2 = nn.Linear(output_dim * 2, output_dim)
        
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(output_dim)
            self.Graph_norm2 = Norm(hidden_dim=output_dim)
        
        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(output_dim)
    
    def forward(self, graph, h, e, injection_mask=None):
        h_in1 = h
        
        if self.batch_norm:
            h = self.Graph_norm1(graph, h)
        
        # SAMP 注意力
        attn_out = self.attention(graph, h, e, injection_mask=injection_mask)
        h = attn_out.view(-1, self.out_channels)
        
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.O(h)
        
        if self.residual:
            h = h_in1 + h
        
        if self.layer_norm:
            h = self.layer_norm1(h)
        
        h_in2 = h
        
        if self.batch_norm:
            h = self.Graph_norm2(graph, h)
        
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)
        
        if self.residual:
            h = h_in2 + h
        
        if self.layer_norm:
            h = self.layer_norm2(h)
        
        return h


# ============================================================
# 5. 改造后的 DevignModel
# ============================================================

class ReparamLargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, small_kernel, large_kernel, stride, groups):
        super().__init__()
        self.large_conv = nn.Conv1d(in_channels, out_channels, kernel_size=large_kernel, 
                                     stride=stride, padding=large_kernel // 2, groups=groups, bias=True)
        self.large_bn = nn.BatchNorm1d(out_channels)
        self.small_conv = nn.Conv1d(in_channels, out_channels, kernel_size=small_kernel, 
                                     stride=stride, padding=small_kernel // 2, groups=groups)
        self.small_bn = nn.BatchNorm1d(out_channels)

    def forward(self, inputs):
        return self.large_bn(self.large_conv(inputs)) + self.small_bn(self.small_conv(inputs))


class EnhancedDevignModel(nn.Module):
    """
    AMPLE + 跨函数语义注入的完整模型。
    
    相比原始 DevignModel 的改动：
    1. 加入 EffectEncoder 和 SummaryInjector
    2. GraphTransformerLayer 替换为 SAMPGraphTransformerLayer
    3. forward 额外接收 injection_mask 和 effect_indices
    """
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super().__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        
        # === 新增：Effect 编码器和注入模块 ===
        self.effect_encoder = EffectEncoder(effect_dim=input_dim)
        self.summary_injector = SummaryInjector(feature_dim=input_dim, effect_dim=input_dim)
        
        # === 输入投影: input_dim -> output_dim ===
        self.input_proj = nn.Linear(input_dim, output_dim)
        
        # === SAMP Graph Transformer ===
        n_layers = 3
        num_head = 10
        self.n_layers = n_layers
        self.gtn = nn.ModuleList([
            SAMPGraphTransformerLayer(
                output_dim, output_dim, num_heads=num_head,
                dropout=0.2, max_edge_types=max_edge_types,
                layer_norm=False, batch_norm=True, residual=True
            ) for i in range(n_layers - 1)
        ])
        
        self.MPL_layer = MLPReadout(output_dim, 2)
        
        # RepLK + ConvFFN (和原始 AMPLE 一样)
        ffn_ratio = 2
        self.concat_dim = output_dim
        small_kernel = 3
        large_kernel = 11
        self.RepLK = nn.Sequential(
            nn.BatchNorm1d(self.concat_dim),
            nn.Conv1d(self.concat_dim, self.concat_dim * ffn_ratio, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            ReparamLargeKernelConv(self.concat_dim * ffn_ratio, self.concat_dim * ffn_ratio, 
                                    small_kernel, large_kernel, stride=1, groups=self.concat_dim * ffn_ratio),
            nn.ReLU(),
            nn.Conv1d(self.concat_dim * ffn_ratio, self.concat_dim, kernel_size=1, stride=1, padding=0),
        )
        k = 3
        self.Avgpool1 = nn.Sequential(nn.ReLU(), nn.AvgPool1d(k, stride=k), nn.Dropout(0.1))
        self.ConvFFN = nn.Sequential(
            nn.BatchNorm1d(self.concat_dim),
            nn.Conv1d(self.concat_dim, self.concat_dim * ffn_ratio, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv1d(self.concat_dim * ffn_ratio, self.concat_dim, kernel_size=1, stride=1, padding=0),
        )
        self.Avgpool2 = nn.Sequential(nn.ReLU(), nn.AvgPool1d(k, stride=k), nn.Dropout(0.1))
    
    def forward(self, batch, cuda=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        graph = graph.to(torch.device('cuda:0'))
        
        # === 获取注入信息 ===
        injection_mask = batch.get_injection_mask(cuda=cuda)
        effect_indices = batch.get_effect_indices()
        
        # === Effect 编码 + 注入 ===
        if injection_mask is not None and injection_mask.sum() > 0:
            effect_vectors = self.effect_encoder(effect_indices)
            features = self.summary_injector(features, effect_vectors, injection_mask)
        
        # === 输入投影 ===
        features = self.input_proj(features)
        
        # === SAMP Graph Transformer ===
        for conv in self.gtn:
            features = conv(graph, features, edge_types, injection_mask=injection_mask)
        
        outputs = batch.de_batchify_graphs(features)
        
        # === RepLK + ConvFFN (和原始 AMPLE 一样) ===
        outputs = outputs.transpose(1, 2)
        outputs += self.RepLK(outputs)
        outputs = self.Avgpool1(outputs)
        outputs += self.ConvFFN(outputs)
        outputs = self.Avgpool2(outputs)
        outputs = outputs.transpose(1, 2)
        
        outputs = self.MPL_layer(outputs.sum(dim=1))
        outputs = nn.Softmax(dim=1)(outputs)
        return outputs


# ============================================================
# 6. 改造后的数据加载器
# ============================================================

class EnhancedDataEntry:
    """
    增强版 DataEntry，额外保存 injection_mask 和 effect_indices。
    """
    def __init__(self, dataset, num_nodes, features, edges, target, 
                 injection_mask=None, effect_indices=None):
        self.dataset = dataset
        self.num_nodes = num_nodes
        self.target = target
        self.graph = DGLGraph()
        self.features = torch.FloatTensor(features)
        self.graph.add_nodes(self.num_nodes, data={'features': self.features})
        for s, _type, t in edges:
            etype_number = self.dataset.get_edge_type_number(_type)
            self.graph.add_edge(s, t, data={'etype': torch.LongTensor([etype_number])})
        
        # 注入信息
        if injection_mask is not None:
            self.injection_mask = torch.FloatTensor(injection_mask)
        else:
            self.injection_mask = torch.zeros(num_nodes)
        
        # effect_indices: 每个节点的 effect 索引列表
        if effect_indices is not None:
            self.effect_indices = effect_indices
        else:
            self.effect_indices = [[] for _ in range(num_nodes)]


class EnhancedBatchGraph:
    """
    增强版 BatchGraph，额外处理 injection_mask 和 effect_indices。
    """
    def __init__(self):
        self.graph = DGLGraph()
        self.number_of_nodes = 0
        self.graphid_to_nodeids = {}
        self.num_of_subgraphs = 0
        self.all_injection_masks = []
        self.all_effect_indices = []
    
    def add_subgraph(self, _g, injection_mask=None, effect_indices=None):
        assert isinstance(_g, DGLGraph)
        num_new_nodes = _g.number_of_nodes()
        
        self.graphid_to_nodeids[self.num_of_subgraphs] = torch.LongTensor(
            list(range(self.number_of_nodes, self.number_of_nodes + num_new_nodes))
        ).to(torch.device('cuda:0'))
        
        self.graph.add_nodes(num_new_nodes, data=_g.ndata)
        sources, dests = _g.all_edges()
        sources += self.number_of_nodes
        dests += self.number_of_nodes
        self.graph.add_edges(sources, dests, data=_g.edata)
        
        # 收集注入信息
        if injection_mask is not None:
            self.all_injection_masks.append(injection_mask)
        else:
            self.all_injection_masks.append(torch.zeros(num_new_nodes))
        
        if effect_indices is not None:
            self.all_effect_indices.extend(effect_indices)
        else:
            self.all_effect_indices.extend([[] for _ in range(num_new_nodes)])
        
        self.number_of_nodes += num_new_nodes
        self.num_of_subgraphs += 1
    
    def get_network_inputs(self, cuda=False, device=None):
        features = self.graph.ndata['features']
        edge_types = self.graph.edata['etype']
        if cuda:
            return self.graph, features.cuda(device=device), edge_types.cuda(device=device)
        return self.graph, features, edge_types
    
    def get_injection_mask(self, cuda=False):
        if not self.all_injection_masks:
            return None
        mask = torch.cat(self.all_injection_masks, dim=0)
        # 如果全为 0，直接返回 None 避免无效计算
        if mask.sum() == 0:
            return None
        if cuda:
            mask = mask.cuda()
        return mask
    
    def get_effect_indices(self):
        return self.all_effect_indices
    
    def de_batchify_graphs(self, features=None):
        assert isinstance(features, torch.Tensor)
        vectors = [features.index_select(dim=0, index=self.graphid_to_nodeids[gid]) 
                   for gid in self.graphid_to_nodeids.keys()]
        lengths = [f.size(0) for f in vectors]
        max_len = max(lengths)
        for i, v in enumerate(vectors):
            vectors[i] = torch.cat(
                (v, torch.zeros(size=(max_len - v.size(0), *(v.shape[1:])), 
                                requires_grad=v.requires_grad, device=v.device)), dim=0)
        return torch.stack(vectors)


class EnhancedDataSet:
    """
    增强版 DataSet，加载 injection_plan。
    """
    def __init__(self, train_src, valid_src, test_src, batch_size, 
                 injection_plan_dir=None,
                 n_ident=None, g_ident=None, l_ident=None):
        self.train_examples = []
        self.valid_examples = []
        self.test_examples = []
        self.train_batches = []
        self.valid_batches = []
        self.test_batches = []
        self.batch_size = batch_size
        self.edge_types = {}
        self.max_etype = 0
        self.feature_size = 0
        self.n_ident, self.g_ident, self.l_ident = load_default_identifiers(n_ident, g_ident, l_ident)
        
        # 加载注入计划
        self.injection_plans = {'train': None, 'test': None, 'valid': None}
        if injection_plan_dir:
            for split in ['train', 'test', 'valid']:
                plan_path = os.path.join(injection_plan_dir, f'{split}_injection_plan.json')
                if os.path.exists(plan_path):
                    debug(f'Loading injection plan: {split}')
                    with open(plan_path) as f:
                        self.injection_plans[split] = json.load(f)
        
        self.read_dataset(train_src, valid_src, test_src)
        self.initialize_dataset()
    
    def initialize_dataset(self):
        self.initialize_train_batch()
        self.initialize_valid_batch()
        self.initialize_test_batch()
    
    def _parse_injection_plan(self, plan):
        """
        把 injection_plan 转换成 injection_mask 和 effect_indices。
        """
        if plan is None:
            return None, None
        
        injection_mask = plan.get('injection_mask', None)
        targets = plan.get('injection_targets', {})
        
        if injection_mask is None:
            return None, None
        
        n_nodes = len(injection_mask)
        effect_indices = [[] for _ in range(n_nodes)]
        
        for node_idx_str, info in targets.items():
            node_idx = int(node_idx_str)
            if node_idx < n_nodes:
                effects = info.get('effects', [])
                for eff in effects:
                    if eff in EFFECT_TO_IDX:
                        effect_indices[node_idx].append(EFFECT_TO_IDX[eff])
        
        return injection_mask, effect_indices
    
    def read_dataset(self, train_src, valid_src, test_src):
        debug('Reading Train File!')
        with open(train_src) as fp:
            train_data = json.load(fp)
            plans = self.injection_plans['train']
            for idx, entry in enumerate(tqdm(train_data)):
                plan = plans[idx] if plans and idx < len(plans) else None
                inj_mask, eff_idx = self._parse_injection_plan(plan)
                
                example = EnhancedDataEntry(
                    dataset=self, 
                    num_nodes=len(entry[self.n_ident]),
                    features=entry[self.n_ident],
                    edges=entry[self.g_ident], 
                    target=entry[self.l_ident][0][0],
                    injection_mask=inj_mask,
                    effect_indices=eff_idx
                )
                if self.feature_size == 0:
                    self.feature_size = example.features.size(1)
                    debug('Feature Size %d' % self.feature_size)
                self.train_examples.append(example)
        
        if valid_src is not None:
            debug('Reading Validation File!')
            with open(valid_src) as fp:
                valid_data = json.load(fp)
                plans = self.injection_plans['valid']
                for idx, entry in enumerate(tqdm(valid_data)):
                    plan = plans[idx] if plans and idx < len(plans) else None
                    inj_mask, eff_idx = self._parse_injection_plan(plan)
                    
                    example = EnhancedDataEntry(
                        dataset=self,
                        num_nodes=len(entry[self.n_ident]),
                        features=entry[self.n_ident],
                        edges=entry[self.g_ident],
                        target=entry[self.l_ident][0][0],
                        injection_mask=inj_mask,
                        effect_indices=eff_idx
                    )
                    self.valid_examples.append(example)
        
        if test_src is not None:
            debug('Reading Test File!')
            with open(test_src) as fp:
                test_data = json.load(fp)
                plans = self.injection_plans['test']
                for idx, entry in enumerate(tqdm(test_data)):
                    plan = plans[idx] if plans and idx < len(plans) else None
                    inj_mask, eff_idx = self._parse_injection_plan(plan)
                    
                    example = EnhancedDataEntry(
                        dataset=self,
                        num_nodes=len(entry[self.n_ident]),
                        features=entry[self.n_ident],
                        edges=entry[self.g_ident],
                        target=entry[self.l_ident][0][0],
                        injection_mask=inj_mask,
                        effect_indices=eff_idx
                    )
                    self.test_examples.append(example)
    
    def get_edge_type_number(self, _type):
        if _type not in self.edge_types:
            self.edge_types[_type] = self.max_etype
            self.max_etype += 1
        return self.edge_types[_type]
    
    @property
    def max_edge_type(self):
        return self.max_etype
    
    def initialize_train_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.train_batches = initialize_batch(self.train_examples, batch_size, shuffle=True)
        return len(self.train_batches)
    
    def initialize_valid_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.valid_batches = initialize_batch(self.valid_examples, batch_size, shuffle=False)
        return len(self.valid_batches)
    
    def initialize_test_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.test_batches = initialize_batch(self.test_examples, batch_size, shuffle=False)
        return len(self.test_batches)
    
    def get_dataset_by_ids_for_GGNN(self, entries, ids):
        taken_entries = [entries[i] for i in ids]
        labels = [e.target for e in taken_entries]
        batch_graph = EnhancedBatchGraph()
        for entry in taken_entries:
            batch_graph.add_subgraph(
                copy.deepcopy(entry.graph),
                injection_mask=entry.injection_mask,
                effect_indices=entry.effect_indices
            )
        return batch_graph, torch.FloatTensor(labels)
    
    def get_next_train_batch(self):
        if len(self.train_batches) == 0:
            self.initialize_train_batch()
        ids = self.train_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.train_examples, ids)
    
    def get_next_valid_batch(self):
        if len(self.valid_batches) == 0:
            self.initialize_valid_batch()
        ids = self.valid_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.valid_examples, ids)
    
    def get_next_test_batch(self):
        if len(self.test_batches) == 0:
            self.initialize_test_batch()
        ids = self.test_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.test_examples, ids)

# ============================================================
# 7. AblationDevignModel — 消融实验模型
# ============================================================

class AblationDevignModel(nn.Module):
    """
    消融实验统一模型。
    
    use_injection=False, use_samp=False  → Baseline (原始 AMPLE)
    use_injection=True,  use_samp=False  → 只注入特征，GNN 不区分两种消息
    use_injection=True,  use_samp=True   → 完整方法（等价于 EnhancedDevignModel）
    """
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8,
                 use_injection=True, use_samp=True):
        super().__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.use_injection = use_injection
        self.use_samp = use_samp

        # Effect 编码 + 注入（可选）
        if use_injection:
            self.effect_encoder = EffectEncoder(effect_dim=input_dim)
            self.summary_injector = SummaryInjector(feature_dim=input_dim, effect_dim=input_dim)

        # GNN 层：use_samp=True 用 SAMP 层，否则用原始 GraphTransformerLayer
        from graph_transformer_layers import GraphTransformerLayer
        n_layers = 3
        num_head = 10
        self.n_layers = n_layers

        # 输入投影: input_dim -> output_dim
        self.input_proj = nn.Linear(input_dim, output_dim)

        if use_samp:
            self.gtn = nn.ModuleList([
                SAMPGraphTransformerLayer(
                    output_dim, output_dim, num_heads=num_head,
                    dropout=0.2, max_edge_types=max_edge_types,
                    layer_norm=False, batch_norm=True, residual=True
                ) for i in range(n_layers - 1)
            ])
        else:
            # 用原始 GraphTransformerLayer（不区分两种消息）
            self.gtn = nn.ModuleList([
                GraphTransformerLayer(
                    output_dim, output_dim, num_heads=num_head,
                    dropout=0.2, max_edge_types=max_edge_types,
                    layer_norm=False, batch_norm=True, residual=True
                ) for i in range(n_layers - 1)
            ])

        self.MPL_layer = MLPReadout(output_dim, 2)

        # RepLK + ConvFFN（和原始 AMPLE 一致）
        ffn_ratio = 2
        self.concat_dim = output_dim
        small_kernel, large_kernel = 3, 11

        self.RepLK = nn.Sequential(
            nn.BatchNorm1d(self.concat_dim),
            nn.Conv1d(self.concat_dim, self.concat_dim * ffn_ratio, 1),
            nn.ReLU(),
            ReparamLargeKernelConv(self.concat_dim * ffn_ratio, self.concat_dim * ffn_ratio,
                                    small_kernel, large_kernel, stride=1,
                                    groups=self.concat_dim * ffn_ratio),
            nn.ReLU(),
            nn.Conv1d(self.concat_dim * ffn_ratio, self.concat_dim, 1),
        )
        k = 3
        self.Avgpool1 = nn.Sequential(nn.ReLU(), nn.AvgPool1d(k, stride=k), nn.Dropout(0.1))
        self.ConvFFN = nn.Sequential(
            nn.BatchNorm1d(self.concat_dim),
            nn.Conv1d(self.concat_dim, self.concat_dim * ffn_ratio, 1),
            nn.GELU(),
            nn.Conv1d(self.concat_dim * ffn_ratio, self.concat_dim, 1),
        )
        self.Avgpool2 = nn.Sequential(nn.ReLU(), nn.AvgPool1d(k, stride=k), nn.Dropout(0.1))

    def forward(self, batch, cuda=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        graph = graph.to(torch.device('cuda:0'))

        # 注入（可选）
        if self.use_injection:
            injection_mask = batch.get_injection_mask(cuda=cuda)
            effect_indices = batch.get_effect_indices()
            if injection_mask is not None and injection_mask.sum() > 0:
                effect_vectors = self.effect_encoder(effect_indices)
                features = self.summary_injector(features, effect_vectors, injection_mask)
        else:
            injection_mask = None

        # GNN 前向
        for conv in self.gtn:
            if self.use_samp:
                features = conv(graph, features, edge_types, injection_mask=injection_mask)
            else:
                features = conv(graph, features, edge_types)

        outputs = batch.de_batchify_graphs(features)
        outputs = outputs.transpose(1, 2)
        outputs += self.RepLK(outputs)
        outputs = self.Avgpool1(outputs)
        outputs += self.ConvFFN(outputs)
        outputs = self.Avgpool2(outputs)
        outputs = outputs.transpose(1, 2)
        outputs = self.MPL_layer(outputs.sum(dim=1))
        return nn.Softmax(dim=1)(outputs)