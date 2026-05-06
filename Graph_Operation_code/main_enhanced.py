#!/usr/bin/env python3
"""
main_enhanced.py — 增强版训练入口

使用方法:
  python main_enhanced.py \
    --dataset vulnsc \
    --input_dir vulnsc_dataset/vulnsc_cpg_c2_2 \
    --injection_plan_dir vulnsc_dataset/vulnsc_injection_plans \
    --feature_size 100 \
    --graph_embed_size 100 \
    --batch_size 64 \
    --epochs 100

对比实验模式 (--ablation):
  baseline       : 原始 AMPLE，不加载注入计划
  node_inject    : 只做特征注入，不用 SAMP（GNN 不区分两种消息）
  no_dataflow    : 不做数据流传播（直接注入 callsite，不做参数级分解）
  full           : 完整方法（默认）
"""

import argparse
import logging
import os
import pickle
import sys
import math

os.chdir(sys.path[0])

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim.optimizer import Optimizer

from trainer import train
from utils import tally_param, debug, set_logger
from enhanced_model import (
    EnhancedDevignModel, EnhancedDataSet,
    AblationDevignModel  # 消融实验模型
)

torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True


# -------------------------------------------------------
# RAdam 优化器（和原 main.py 一致）
# -------------------------------------------------------
class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        beta2_t = ratio = N_sma_max = N_sma = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1
                if beta2_t is None:
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    beta1_t = 1 - beta1 ** state['step']
                    if N_sma >= 5:
                        ratio = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4)
                            * (N_sma - 2) / N_sma
                            * N_sma_max / (N_sma_max - 2)
                        ) / beta1_t

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])

                if N_sma >= 5:
                    step_size = group['lr'] * ratio
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    step_size = group['lr'] / beta1_t
                    p_data_fp32.add_(exp_avg, alpha=-step_size)

                p.data.copy_(p_data_fp32)
        return loss


# -------------------------------------------------------
# 消融实验：决定加载哪种模型
# -------------------------------------------------------
def build_model(ablation, input_dim, output_dim, max_edge_types, num_steps):
    """
    ablation 模式说明：
      'full'         : 完整方法（EffectEncoder + SummaryInjector + SAMP）
      'no_samp'      : 只注入特征，GNN 不区分两种消息（消融 SAMP）
      'no_dataflow'  : 整条摘要注入 callsite，不做参数级分解（消融数据流传播）
      'baseline'     : 原始 AMPLE，不做任何注入
    """
    if ablation == 'baseline':
        # 使用消融模型，禁用所有注入
        return AblationDevignModel(
            input_dim=input_dim, output_dim=output_dim,
            max_edge_types=max_edge_types, num_steps=num_steps,
            use_injection=False, use_samp=False
        )
    elif ablation == 'no_samp':
        # 有注入，但 GNN 不区分两种消息
        return AblationDevignModel(
            input_dim=input_dim, output_dim=output_dim,
            max_edge_types=max_edge_types, num_steps=num_steps,
            use_injection=True, use_samp=False
        )
    elif ablation == 'no_dataflow':
        # 有 SAMP，但注入计划来自"无数据流传播"版本
        # (注入计划在预处理阶段生成，这里模型不变，只改数据)
        return EnhancedDevignModel(
            input_dim=input_dim, output_dim=output_dim,
            max_edge_types=max_edge_types, num_steps=num_steps
        )
    else:  # 'full'
        return EnhancedDevignModel(
            input_dim=input_dim, output_dim=output_dim,
            max_edge_types=max_edge_types, num_steps=num_steps
        )


# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == '__main__':
    torch.manual_seed(10)
    np.random.seed(10)

    parser = argparse.ArgumentParser(description='Enhanced AMPLE with cross-function summary injection')
    parser.add_argument('--dataset', type=str, default='vulnsc')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='CPG 图数据目录，包含 vulnsc-train-v0.json 等')
    parser.add_argument('--injection_plan_dir', type=str, default=None,
                        help='注入计划目录，包含 train/test/valid_injection_plan.json')
    parser.add_argument('--log_dir', type=str, default='enhanced.log')
    parser.add_argument('--node_tag', type=str, default='node_features')
    parser.add_argument('--graph_tag', type=str, default='graph')
    parser.add_argument('--label_tag', type=str, default='targets')
    parser.add_argument('--feature_size', type=int, default=100,
                        help='节点特征维度（word2vec）')
    parser.add_argument('--graph_embed_size', type=int, default=100,
                        help='GNN 输出维度')
    parser.add_argument('--num_steps', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_patience', type=int, default=100)
    parser.add_argument('--ablation', type=str, default='full',
                        choices=['full', 'no_samp', 'no_dataflow', 'baseline'],
                        help='消融实验模式')
    args = parser.parse_args()

    # 目录设置
    model_dir = os.path.join('models', f'{args.dataset}_{args.ablation}')
    os.makedirs(model_dir, exist_ok=True)
    log_path = os.path.join(model_dir, args.log_dir)
    set_logger(log_path)

    logging.info('=' * 100)
    logging.info(f'Ablation mode: {args.ablation}')
    logging.info(f'Injection plan dir: {args.injection_plan_dir}')
    logging.info(f'Feature size: {args.feature_size}, Graph embed size: {args.graph_embed_size}')
    logging.info('=' * 100)

    if args.feature_size > args.graph_embed_size:
        logging.warning('graph_embed_size < feature_size, 自动对齐')
        args.graph_embed_size = args.feature_size

    # -------------------------------------------------------
    # 数据加载
    # -------------------------------------------------------
    input_dir = args.input_dir
    processed_data_path = os.path.join(input_dir, f'enhanced_{args.ablation}.bin')

    if os.path.exists(processed_data_path):
        debug(f'Loading cached dataset from {processed_data_path}')
        dataset = pickle.load(open(processed_data_path, 'rb'))
    else:
        # baseline 模式不加载注入计划
        plan_dir = None if args.ablation == 'baseline' else args.injection_plan_dir

        dataset = EnhancedDataSet(
            train_src=os.path.join(input_dir, 'vulnsc-train-v0.json'),
            valid_src=os.path.join(input_dir, 'vulnsc-valid-v0.json'),
            test_src=os.path.join(input_dir, 'vulnsc-test-v0.json'),
            batch_size=args.batch_size,
            injection_plan_dir=plan_dir,
            n_ident=args.node_tag,
            g_ident=args.graph_tag,
            l_ident=args.label_tag
        )
        with open(processed_data_path, 'wb') as f:
            pickle.dump(dataset, f)

    logging.info(f'Train: {len(dataset.train_examples)}, '
                 f'Valid: {len(dataset.valid_examples)}, '
                 f'Test: {len(dataset.test_examples)}')
    logging.info(f'Train batches: {len(dataset.train_batches)}, '
                 f'Valid batches: {len(dataset.valid_batches)}, '
                 f'Test batches: {len(dataset.test_batches)}')

    assert args.feature_size == dataset.feature_size, (
        f'特征维度不一致：参数 {args.feature_size} vs 数据集 {dataset.feature_size}'
    )

    # -------------------------------------------------------
    # 模型构建
    # -------------------------------------------------------
    model = build_model(
        ablation=args.ablation,
        input_dim=dataset.feature_size,
        output_dim=args.graph_embed_size,
        max_edge_types=dataset.max_edge_type,
        num_steps=args.num_steps
    )

    debug(f'Total Parameters: {tally_param(model):,}')
    logging.info(f'Total Parameters: {tally_param(model):,}')
    logging.info(f'Model: {model.__class__.__name__}')

    model.cuda()
    loss_function = CrossEntropyLoss(
        weight=torch.from_numpy(np.array([1, 1.2])).float(),
        reduction='mean'
    ).cuda()

    optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    # -------------------------------------------------------
    # 训练
    # -------------------------------------------------------
    train(
        model=model,
        dataset=dataset,
        epoches=args.epochs,
        dev_every=len(dataset.train_batches),
        loss_function=loss_function,
        optimizer=optimizer,
        save_path=os.path.join(model_dir, 'EnhancedModel'),
        max_patience=args.max_patience,
        log_every=5
    )