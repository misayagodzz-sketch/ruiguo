"""
code_summary.py

对 code_retrieval_vulnsc.py 输出的 callee 源码，用 DeepSeek 生成结构化安全语义摘要。
基于 VulnSC 原版改写，加入：
  - 安全语义标签（security_effects + security_score）
  - async 并发 + 断点续传
  - 4种 prompt: basic(0) / behavior_guide(1) / oneshot(2) / cot(3)
  - --limit 参数方便调试

用法：
    python code_summary.py \
        --input   /path/to/devign_source.jsonl \
        --out_dir /path/to/vuldata/summary \
        --api_key sk-xxxx \
        --model   deepseek-chat \
        --ptid    0 \
        --concurrency 8 \
        --limit   10
"""

import os
import re
import json
import asyncio
import argparse
import aiohttp
from tqdm import tqdm


# ── 安全语义标签 ──────────────────────────────────────────────────────────────

SECURITY_EFFECTS = [
    "NULL_RET",          # 返回值可能为NULL，调用方需检查
    "ALLOC_RET",         # 返回堆内存，调用方负责释放
    "FREE_EFFECT",       # 释放传入的内存
    "WRITE_BUFFER",      # 向缓冲区写数据，存在越界风险
    "READ_BUFFER",       # 从缓冲区读数据
    "LEN_DEPENDENT",     # 操作长度/大小参数，存在整数溢出风险
    "CHECK_REQUIRED",    # 返回值必须检查（错误码/状态码）
    "INIT_STATE",        # 初始化数据结构状态
    "RESOURCE_ACQUIRE",  # 获取系统资源（锁/fd/句柄）
    "RESOURCE_RELEASE",  # 释放系统资源
]

PTID_NAME = {0: "basic", 1: "behavior_guide", 2: "oneshot", 3: "cot"}

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an advanced code summarization tool specialized in security analysis.\n"
    "Given the source code of a C function, generate a structured summary.\n"
    "Provide response in the following format:\n"
    "input:<INPUTS> | output:<OUTPUTS> | behavior:<BEHAVIOR> | "
    "effects:<EFFECT1,EFFECT2> | score:<0.0-1.0>\n\n"
    "Where:\n"
    "- input: parameters the function accepts\n"
    "- output: return values of the function (not the type)\n"
    "- behavior: concisely describes what the function does with input/output variables\n"
    f"- effects: comma-separated subset of {', '.join(SECURITY_EFFECTS)}, or NONE\n"
    "- score: security relevance float (0.0=none, 0.5=moderate memory/resource ops, 1.0=direct vuln pattern)\n"
    "No extra text. No markdown."
)

# ── 4种 User prompt ───────────────────────────────────────────────────────────

def user_prompt_basic(source_code):
    return (
        "Only provide the response in the format mentioned before "
        "for the following code snippet:\n" + source_code
    )

def user_prompt_behavior_guide(source_code):
    guide = (
        "The behaviors should relate to these security-relevant categories:\n"
        "1. Access Control: managing permissions\n"
        "2. Arithmetic Operations: calculations with overflow risk\n"
        "3. Authentication: verifying identities\n"
        "4. Code Execution: executing code or commands\n"
        "5. Concurrency: managing threads or processes\n"
        "6. File Handling: reading or writing files\n"
        "7. Input Validation: checking or sanitizing inputs\n"
        "8. Memory Management: allocating or freeing memory\n"
        "9. Network Communication: sending or receiving data\n"
        "10. Path Management: managing file or directory paths\n"
        "Only describe the behavior concisely without naming the category.\n\n"
    )
    return (
        guide +
        "Only provide the response in the format mentioned before "
        "for the following code snippet:\n" + source_code
    )

def user_prompt_oneshot(source_code):
    example = (
        "Given an example as follow:\n"
        "char* safe_strcpy(char *dst, const char *src, size_t size) {\n"
        "    if (!dst || !src || size == 0) return NULL;\n"
        "    strncpy(dst, src, size - 1);\n"
        "    dst[size-1] = '\\0';\n"
        "    return dst;\n"
        "}\n\n"
        "The summary is: "
        "input:char *dst, const char *src, size_t size | "
        "output:char* dst on success or NULL on invalid input | "
        "behavior:Copies src into dst with size bound, null-terminates result, returns NULL if inputs are invalid | "
        "effects:NULL_RET,WRITE_BUFFER,LEN_DEPENDENT | score:0.7\n\n"
    )
    return (
        example +
        "Only provide the response in the format mentioned before "
        "for the following code snippet:\n" + source_code
    )

def user_prompt_cot(source_code):
    return (
        "Let us think step by step. "
        "Only provide the response in the format mentioned before "
        "for the following code snippet:\n" + source_code
    )

PROMPT_FNS = {
    0: user_prompt_basic,
    1: user_prompt_behavior_guide,
    2: user_prompt_oneshot,
    3: user_prompt_cot,
}

# ── 解析 LLM 输出 ─────────────────────────────────────────────────────────────

def parse_response(text):
    """解析 LLM 返回的 pipe-separated 格式，返回 dict 或 None。"""
    try:
        fields = {}
        for part in re.split(r'\s*\|\s*', text.strip()):
            if ':' in part:
                key, _, val = part.partition(':')
                fields[key.strip().lower()] = val.strip()

        if not all(k in fields for k in ['input', 'output', 'behavior']):
            return None

        # 解析 effects
        effects_raw = fields.get('effects', 'NONE')
        if effects_raw.upper() == 'NONE':
            effects = []
        else:
            effects = [e.strip() for e in effects_raw.split(',')
                       if e.strip() in SECURITY_EFFECTS]

        # 解析 score
        score_raw = fields.get('score', '0.0')
        m = re.search(r'[\d.]+', score_raw)
        score = min(1.0, max(0.0, float(m.group()))) if m else 0.0

        return {
            'input':            fields['input'],
            'output':           fields['output'],
            'behavior':         fields['behavior'],
            'security_effects': effects,
            'security_score':   score,
        }
    except Exception:
        return None

# ── async API 调用 ────────────────────────────────────────────────────────────

async def call_api(session, api_key, model, func_name, func_str, ptid, semaphore, max_retries=3):
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    user_content = PROMPT_FNS[ptid](func_str)
    user_content = user_content[:7000]  # 与原版保持一致

    payload = {
        "model": model,
        "max_tokens": 300,
        "temperature": 0.0,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
    }

    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.post(
                    url, headers=headers, json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    if resp.status != 200:
                        return None
                    data = await resp.json()
                    text = data["choices"][0]["message"]["content"]
                    parsed = parse_response(text)
                    if parsed is None:
                        return None
                    parsed['func_name'] = func_name
                    parsed['raw'] = text  # 保留原始输出方便调试
                    return parsed
            except Exception:
                await asyncio.sleep(2 ** attempt)
    return None

# ── 主流程 ────────────────────────────────────────────────────────────────────

async def main_async(args):
    prompt_name = PTID_NAME[args.ptid]
    out_path = os.path.join(args.out_dir, args.model, prompt_name, 'callee_summaries.jsonl')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 断点续传：加载已完成的
    done = {}
    if os.path.exists(out_path):
        with open(out_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    done[item['func_name']] = item
                except Exception:
                    pass
        print(f"断点续传：已有 {len(done)} 个函数摘要")

    # 收集所有唯一 callee（取最长源码）
    func_map = {}
    count = 0
    with open(args.input, 'r') as f:
        for line in f:
            item = json.loads(line)
            for callee in item.get('callee', []):
                fname = callee['func_name']
                fstr  = callee['func_str']
                if fname not in func_map or len(fstr) > len(func_map[fname]):
                    func_map[fname] = fstr
            count += 1
            if args.limit and count >= args.limit:
                break

    todo = {k: v for k, v in func_map.items() if k not in done}
    print(f"总唯一函数: {len(func_map)}, 已完成: {len(done)}, 待处理: {len(todo)}")

    if not todo:
        print("全部完成！")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    connector  = aiohttp.TCPConnector(limit=args.concurrency * 2)

    success = fail = 0
    async with aiohttp.ClientSession(connector=connector) as session:
        with open(out_path, 'a') as out_f:
            tasks = [
                call_api(session, args.api_key, args.model, fname, fstr, args.ptid, semaphore)
                for fname, fstr in todo.items()
            ]
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"[{args.model}/{prompt_name}]"):
                result = await coro
                if result is not None:
                    out_f.write(json.dumps(result) + '\n')
                    out_f.flush()
                    success += 1
                else:
                    fail += 1

    print(f"\n完成：success={success}, fail={fail}")
    print(f"输出：{out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',       required=True,           help='devign_source.jsonl 路径')
    parser.add_argument('--out_dir',     required=True,           help='输出根目录，如 vuldata/summary')
    parser.add_argument('--api_key',     required=True,           help='DeepSeek API key')
    parser.add_argument('--model',       default='deepseek-chat', help='模型名')
    parser.add_argument('--ptid',        type=int, default=0,     help='prompt类型: 0=basic 1=behavior_guide 2=oneshot 3=cot')
    parser.add_argument('--concurrency', type=int, default=8,     help='并发数')
    parser.add_argument('--limit',       type=int, default=None,  help='只处理前N条样本（调试用）')
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()