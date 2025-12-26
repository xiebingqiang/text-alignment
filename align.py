#!/usr/bin/env python3
"""
双语文本对齐工具
输入两个文本文件，输出对齐好的CSV
"""

import os
import re
import argparse
import pandas as pd
import requests
import time
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ============ 配置参数 ============
DEFAULT_MIN_LINE_LENGTH = 200
DEFAULT_WINDOW_SIZE = 0.01
DEFAULT_OUTPUT_CSV = 'aligned_result.csv'
DEFAULT_CHECKPOINT_FILE = 'checkpoint.json'
MAX_RETRIES = 3
RETRY_DELAY = 1
DEFAULT_THREAD_COUNT = 30
DEBUG_SAMPLE = 2
CHECKPOINT_INTERVAL = 10

# API配置
DEFAULT_API_URL = 'https://api.lkeap.cloud.tencent.com/v1'
DEFAULT_MODEL = 'deepseek-v3'

# ============ 工具函数 ============

def load_api_keys():
    """从多个来源加载API密钥"""
    env_keys = os.environ.get('ALIYUN_DEEPSEEK_API_KEYS', '')
    if env_keys:
        keys = [k.strip() for k in env_keys.split(',') if k.strip()]
        if keys:
            return keys

    keys_file = os.path.join(os.path.dirname(__file__) or '.', 'api_keys.txt')
    if os.path.exists(keys_file):
        with open(keys_file, 'r') as f:
            keys = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            if keys:
                return keys

    config_file = os.path.join(os.path.dirname(__file__) or '.', 'config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            keys = config.get('api_keys', [])
            if keys:
                return keys

    return []


def _is_cjk_text(text):
    """判断文本是否主要是中日韩文字"""
    if not text:
        return False
    cjk_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff' or
                    '\u3040' <= char <= '\u309f' or
                    '\u30a0' <= char <= '\u30ff' or
                    '\uac00' <= char <= '\ud7af')
    return cjk_count > len(text) * 0.3


# ============ 第一阶段：位置法粗对齐 ============

def preprocess_text(text):
    """通用文本预处理"""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    processed_paragraphs = []

    for para in paragraphs:
        lines = [line.strip() for line in para.split('\n') if line.strip()]
        merged_lines = []
        buffer = ""

        for line in lines:
            is_heading = line.isupper() and 2 <= len(line.split()) <= 5

            should_merge = (
                not is_heading
                and not re.search(r'[.!?。！？][""]?\s*\d*$', buffer)
                and not (buffer.endswith('"') and len(line) > 0 and line[0].isupper())
                and not (buffer and buffer[-1].isupper() and len(line) > 0 and line[0].isupper())
            )

            if should_merge:
                if buffer.endswith('-'):
                    separator = ''
                elif _is_cjk_text(buffer) or _is_cjk_text(line):
                    separator = ''
                else:
                    separator = ' '
                buffer += separator + line
            else:
                if buffer:
                    merged_lines.append(buffer)
                buffer = line

        if buffer:
            merged_lines.append(buffer)

        refined_para = []
        for line in merged_lines:
            line = re.sub(r'([""])\s*(\d+)', r'\1\2', line)
            refined_para.append(line)

        processed_paragraphs.append('\n'.join(refined_para))

    return '\n\n'.join(processed_paragraphs)


def merge_short_paragraphs(text, min_length):
    """合并短段落直到达到最小长度"""
    paragraphs = text.split('\n')
    merged = []
    current_para = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if not current_para:
            current_para = para
        else:
            current_para += "\n" + para

        if len(current_para) >= min_length or para == paragraphs[-1]:
            merged.append(current_para)
            current_para = ""

    return merged


def calculate_ranges(full_text, paragraphs):
    """计算合并后段落在全文中的位置范围"""
    total_len = len(full_text)
    ranges = []

    for para in paragraphs:
        start = full_text.find(para)
        if start == -1:
            continue
        end = start + len(para)
        ranges.append((para, start/total_len, end/total_len))

    return ranges


def rough_align(text1, text2, min_length, window_size):
    """第一阶段：位置法粗对齐

    用text1做分段基准，在text2里找对应内容
    """
    text2_processed = preprocess_text(text2)
    text1_cleaned = '\n'.join([line.strip() for line in text1.split('\n') if line.strip()])
    text1_paragraphs = merge_short_paragraphs(text1_cleaned, min_length)
    text1_ranges = calculate_ranges(text1_cleaned, text1_paragraphs)

    results = []
    for para, start_pct, end_pct in text1_ranges:
        # 按比例在text2里找对应区间
        text2_start = max(0, int((start_pct - window_size*2) * len(text2_processed)))
        text2_end = min(len(text2_processed), int((end_pct + window_size) * len(text2_processed)))

        results.append({
            'text_a': para,  # 来自file1的段落
            'text_b': text2_processed[text2_start:text2_end],  # file2的候选区
        })

    return results


# ============ 第二阶段：AI精确对齐 ============

class KeyManager:
    """密钥轮询管理"""
    def __init__(self, keys):
        self.keys = keys
        self.lock = threading.Lock()
        self.key_usage = {k: {'count': 0, 'last_used': 0} for k in keys}

    def get_key(self):
        with self.lock:
            key = min(self.key_usage.items(),
                      key=lambda x: (x[1]['count'], x[1]['last_used']))[0]
            self.key_usage[key]['count'] += 1
            self.key_usage[key]['last_used'] = time.time()
            return key


class CheckpointManager:
    """断点续传管理器"""
    def __init__(self, checkpoint_file, output_csv):
        self.checkpoint_file = checkpoint_file
        self.output_csv = output_csv
        self.lock = threading.Lock()
        self.completed_indices = set()
        self.results = {}
        self.save_counter = 0

    def load(self):
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.completed_indices = set(data.get('completed_indices', []))
                    self.results = {int(k): v for k, v in data.get('results', {}).items()}
                    print(f"从检查点恢复: 已完成 {len(self.completed_indices)} 条")
                    return True
            except Exception as e:
                print(f"加载检查点失败: {e}")

        # 兼容旧版输出文件
        if os.path.exists(self.output_csv):
            try:
                df = pd.read_csv(self.output_csv)
                if '_index' in df.columns:
                    for _, row in df.iterrows():
                        idx = int(row['_index'])
                        self.completed_indices.add(idx)
                        # 兼容新旧字段名
                        text_a = row.get('text_a') or row.get('target_text', '')
                        text_b = row.get('text_b') or row.get('source_text', '')
                        self.results[idx] = {'text_a': text_a, 'text_b': text_b}
                    print(f"从输出文件恢复: 已完成 {len(self.completed_indices)} 条")
                    return True
            except Exception:
                pass

        return False

    def mark_completed(self, index, text_a, text_b):
        with self.lock:
            self.completed_indices.add(index)
            self.results[index] = {
                'text_a': text_a,
                'text_b': text_b
            }
            self.save_counter += 1
            if self.save_counter >= CHECKPOINT_INTERVAL:
                self._save_checkpoint()
                self.save_counter = 0

    def _save_checkpoint(self):
        try:
            checkpoint_data = {
                'completed_indices': list(self.completed_indices),
                'results': self.results,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

            if self.results:
                rows = []
                for idx in sorted(self.results.keys()):
                    r = self.results[idx]
                    rows.append({
                        '_index': idx,
                        'text_a': r.get('text_a', ''),
                        'text_b': r.get('text_b', '')
                    })
                pd.DataFrame(rows).to_csv(self.output_csv, index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"保存检查点失败: {e}")

    def save_final(self):
        with self.lock:
            self._save_checkpoint()

    def get_pending_indices(self, total):
        with self.lock:
            return [i for i in range(total) if i not in self.completed_indices]

    def get_results_count(self):
        with self.lock:
            return len(self.completed_indices)


# AI自动识别语言和对应关系
SYSTEM_PROMPT = """你是一个专业的双语对齐助手。请严格遵循以下规则：
1. 给你两段不同语言的文本，它们互为翻译关系
2. 文本A较短，文本B较长且包含A对应的内容以及一些多余内容
3. 请在文本B中找出与文本A对应的段落
4. 如果找不到对应内容，返回"无对应内容"
5. 注意不要遗漏小标题，注解序号请包裹在[]号中
6. 只返回匹配的内容，不要添加任何解释
"""


def query_api(prompt, key_manager, api_url, model, retries=MAX_RETRIES):
    """调用API"""
    for attempt in range(retries):
        api_key = key_manager.get_key()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "stream": False
        }

        try:
            response = requests.post(
                f"{api_url}/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=(15, 30)
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"密钥 {api_key[-4:]} 达到限制，切换密钥...")
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            raise
        except Exception as e:
            print(f"请求失败(尝试{attempt+1}/{retries}): {str(e)}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            return "API_ERROR"


def align_pair(args, key_manager, api_url, model):
    """对齐单个配对"""
    index, text_a, text_b = args
    try:
        prompt = f"文本A:\n{text_a}\n\n文本B:\n{text_b}"
        aligned = query_api(prompt, key_manager, api_url, model)

        if aligned == "API_ERROR":
            return (index, text_a, "API请求失败")
        elif "无对应内容" in aligned or not aligned:
            return (index, text_a, "无对应内容")
        return (index, text_a, aligned)
    except Exception as e:
        print(f"处理第{index}条出错: {str(e)}")
        return (index, text_a, "处理错误")


def debug_sample(pairs, key_manager, api_url, model, sample_count):
    """调试模式：处理几条样本看效果"""
    print(f"\n=== 调试模式(前{sample_count}个样本) ===")

    for i, pair in enumerate(pairs[:sample_count]):
        text_a = pair['text_a']
        text_b = pair['text_b']
        start_time = time.time()
        result = align_pair((i, text_a, text_b), key_manager, api_url, model)
        elapsed = time.time() - start_time

        print(f"\n样本{i+1} (耗时:{elapsed:.2f}s):")
        print(f"文本A: {result[1][:100]}..." if len(result[1]) > 100 else f"文本A: {result[1]}")
        print(f"对齐结果: {result[2][:100]}..." if len(result[2]) > 100 else f"对齐结果: {result[2]}")
        time.sleep(0.5)


def ai_align(pairs, key_manager, api_url, model, output_csv, checkpoint_file,
             thread_count, resume=True):
    """第二阶段：AI精确对齐"""
    checkpoint = CheckpointManager(checkpoint_file, output_csv)

    if resume:
        checkpoint.load()

    pending_indices = checkpoint.get_pending_indices(len(pairs))

    if not pending_indices:
        print("所有任务已完成！")
        return

    completed_before = checkpoint.get_results_count()
    print(f"总任务: {len(pairs)}, 已完成: {completed_before}, 待处理: {len(pending_indices)}")

    failed_count = 0

    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        tasks = [(i, pairs[i]['text_a'], pairs[i]['text_b'])
                 for i in pending_indices]

        futures = {
            executor.submit(align_pair, task, key_manager, api_url, model): task[0]
            for task in tasks
        }

        try:
            for future in tqdm(as_completed(futures), total=len(tasks), desc="AI对齐进度"):
                idx = futures[future]
                try:
                    _, text_a, aligned_b = future.result()
                    checkpoint.mark_completed(idx, text_a, aligned_b)
                except Exception as e:
                    print(f"任务{idx}异常: {str(e)}")
                    failed_count += 1
        except KeyboardInterrupt:
            print("\n\n检测到中断信号，正在保存进度...")
            checkpoint.save_final()
            completed_now = checkpoint.get_results_count()
            print(f"进度已保存！本次完成: {completed_now - completed_before} 条")
            print(f"下次运行将自动继续")
            raise

    checkpoint.save_final()

    final_count = checkpoint.get_results_count()
    print(f"\n处理完成！总计: {final_count} 条, 本次新增: {final_count - completed_before} 条, 失败: {failed_count} 条")
    print(f"结果已保存到 {output_csv}")


# ============ 主程序 ============

def main():
    parser = argparse.ArgumentParser(
        description='双语文本对齐工具 - 输入两个文本文件，输出对齐好的CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python align.py english.txt chinese.txt
  python align.py english.txt chinese.txt -o result.csv
  python align.py file1.txt file2.txt --skip-ai  # 仅粗对齐

API密钥配置方式：
  1. 环境变量: export ALIYUN_DEEPSEEK_API_KEYS="key1,key2,key3"
  2. 配置文件: api_keys.txt（每行一个密钥）
        '''
    )

    parser.add_argument('file1', help='第一个文本文件')
    parser.add_argument('file2', help='第二个文本文件')
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT_CSV,
                        help=f'输出CSV文件路径（默认: {DEFAULT_OUTPUT_CSV}）')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT_FILE,
                        help=f'检查点文件路径（默认: {DEFAULT_CHECKPOINT_FILE}）')
    parser.add_argument('--min-length', type=int, default=DEFAULT_MIN_LINE_LENGTH,
                        help=f'最小段落长度（默认: {DEFAULT_MIN_LINE_LENGTH}）')
    parser.add_argument('--window-size', type=float, default=DEFAULT_WINDOW_SIZE,
                        help=f'窗口冗余量（默认: {DEFAULT_WINDOW_SIZE}）')
    parser.add_argument('--encoding', default='utf-8',
                        help='文件编码（默认: utf-8）')
    parser.add_argument('--api-url', default=DEFAULT_API_URL,
                        help=f'API URL（默认: {DEFAULT_API_URL}）')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help=f'模型名称（默认: {DEFAULT_MODEL}）')
    parser.add_argument('--threads', type=int, default=DEFAULT_THREAD_COUNT,
                        help=f'线程数（默认: {DEFAULT_THREAD_COUNT}）')
    parser.add_argument('--debug-samples', type=int, default=DEBUG_SAMPLE,
                        help=f'调试样本数（默认: {DEBUG_SAMPLE}）')
    parser.add_argument('--skip-debug', action='store_true',
                        help='跳过调试模式，直接处理')
    parser.add_argument('--skip-ai', action='store_true',
                        help='跳过AI对齐，仅输出粗对齐结果')
    parser.add_argument('--no-resume', action='store_true',
                        help='不恢复进度，从头开始')

    args = parser.parse_args()

    # 读取文件
    print("读取文件...")
    try:
        with open(args.file1, 'r', encoding=args.encoding) as f:
            text1 = f.read()
        with open(args.file2, 'r', encoding=args.encoding) as f:
            text2 = f.read()
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e.filename}")
        return 1
    except UnicodeDecodeError:
        print(f"错误: 文件编码错误，请尝试指定 --encoding 参数")
        return 1

    print(f"文件1: {len(text1)} 字符")
    print(f"文件2: {len(text2)} 字符")

    # 第一阶段：粗对齐
    print("\n第一阶段：位置法粗对齐...")
    pairs = rough_align(text1, text2, args.min_length, args.window_size)
    print(f"粗对齐完成，共 {len(pairs)} 个段落")

    # 仅粗对齐模式
    if args.skip_ai:
        df = pd.DataFrame(pairs)
        df.to_csv(args.output, index=False, encoding='utf-8-sig')
        print(f"粗对齐结果已保存到 {args.output}")
        return 0

    # 加载API密钥
    api_keys = load_api_keys()
    if not api_keys:
        print("\n错误: 未找到API密钥！")
        print("请通过以下方式之一配置API密钥：")
        print("  1. 环境变量: export ALIYUN_DEEPSEEK_API_KEYS='key1,key2,key3'")
        print("  2. 创建文件: api_keys.txt（每行一个密钥）")
        return 1

    print(f"\n已加载 {len(api_keys)} 个API密钥")
    key_manager = KeyManager(api_keys)

    # 调试模式
    if not args.skip_debug:
        debug_sample(pairs, key_manager, args.api_url, args.model, args.debug_samples)

        response = input("\n是否继续处理完整文件？(y/n): ")
        if response.lower() != 'y':
            print("已取消处理")
            return 0

    # 第二阶段：AI精确对齐
    print("\n第二阶段：AI精确对齐...")
    try:
        ai_align(pairs, key_manager, args.api_url, args.model,
                 args.output, args.checkpoint, args.threads,
                 resume=not args.no_resume)
    except KeyboardInterrupt:
        print("\n程序已中断，进度已保存")
        return 0

    return 0


if __name__ == "__main__":
    exit(main())
