import os
import argparse
import pandas as pd
import requests
import time
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 配置参数
DEFAULT_INPUT_CSV = 'aligned_texts.csv'
DEFAULT_OUTPUT_CSV = 'aligned_final_multithread.csv'
DEFAULT_CHECKPOINT_FILE = 'checkpoint.json'
MAX_RETRIES = 3
RETRY_DELAY = 1
DEFAULT_THREAD_COUNT = 30
DEBUG_SAMPLE = 2
CHECKPOINT_INTERVAL = 10  # 每处理N条保存一次检查点

# API配置
DEFAULT_API_URL = 'https://api.lkeap.cloud.tencent.com/v1'
DEFAULT_MODEL = 'deepseek-v3'


def load_api_keys():
    """
    从多个来源加载API密钥，优先级：
    1. 环境变量 ALIYUN_DEEPSEEK_API_KEYS（逗号分隔）
    2. 配置文件 api_keys.txt（每行一个密钥）
    3. 配置文件 config.json 中的 api_keys 字段
    """
    # 尝试从环境变量读取
    env_keys = os.environ.get('ALIYUN_DEEPSEEK_API_KEYS', '')
    if env_keys:
        keys = [k.strip() for k in env_keys.split(',') if k.strip()]
        if keys:
            return keys

    # 尝试从 api_keys.txt 读取
    keys_file = os.path.join(os.path.dirname(__file__) or '.', 'api_keys.txt')
    if os.path.exists(keys_file):
        with open(keys_file, 'r') as f:
            keys = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            if keys:
                return keys

    # 尝试从 config.json 读取
    config_file = os.path.join(os.path.dirname(__file__) or '.', 'config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            keys = config.get('api_keys', [])
            if keys:
                return keys

    return []


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
        """加载已有的检查点"""
        # 从检查点文件加载
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

        # 从已有的输出文件加载
        if os.path.exists(self.output_csv):
            try:
                df = pd.read_csv(self.output_csv)
                if '_index' in df.columns:
                    for _, row in df.iterrows():
                        idx = int(row['_index'])
                        self.completed_indices.add(idx)
                        self.results[idx] = {
                            'target_text': row['target_text'],
                            'source_text': row['source_text']
                        }
                    print(f"从输出文件恢复: 已完成 {len(self.completed_indices)} 条")
                    return True
            except Exception as e:
                print(f"加载输出文件失败: {e}")

        return False

    def is_completed(self, index):
        """检查某条是否已完成"""
        with self.lock:
            return index in self.completed_indices

    def mark_completed(self, index, target_text, source_text):
        """标记某条为已完成"""
        with self.lock:
            self.completed_indices.add(index)
            self.results[index] = {
                'target_text': target_text,
                'source_text': source_text
            }
            self.save_counter += 1

            # 每N条保存一次
            if self.save_counter >= CHECKPOINT_INTERVAL:
                self._save_checkpoint()
                self.save_counter = 0

    def _save_checkpoint(self):
        """保存检查点（内部方法，需要在锁内调用）"""
        try:
            # 保存检查点JSON
            checkpoint_data = {
                'completed_indices': list(self.completed_indices),
                'results': self.results,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

            # 同时保存CSV（带索引）
            if self.results:
                rows = []
                for idx in sorted(self.results.keys()):
                    row = self.results[idx].copy()
                    row['_index'] = idx
                    rows.append(row)
                pd.DataFrame(rows).to_csv(self.output_csv, index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"保存检查点失败: {e}")

    def save_final(self):
        """保存最终结果"""
        with self.lock:
            self._save_checkpoint()
            # 删除检查点文件（可选）
            # if os.path.exists(self.checkpoint_file):
            #     os.remove(self.checkpoint_file)

    def get_pending_indices(self, total):
        """获取待处理的索引列表"""
        with self.lock:
            return [i for i in range(total) if i not in self.completed_indices]

    def get_results_count(self):
        """获取已完成的数量"""
        with self.lock:
            return len(self.completed_indices)


def get_system_prompt(source_lang=None, target_lang=None):
    """
    生成系统提示词

    Args:
        source_lang: 源语言名称（如 "英文"、"English"）
        target_lang: 目标语言名称（如 "中文"、"Chinese"）
    """
    source_name = source_lang or "源语言"
    target_name = target_lang or "目标语言"

    return f"""你是一个专业的双语对齐助手。请严格遵循以下规则：
1. 给你一段{source_name}原文和{target_name}译文，其中译文是由原文译出
2. 请在{source_name}原文中找出与{target_name}译文对应的段落
3. 如果原文中没有对应内容，返回"无对应内容"
4. 请注意不要遗漏小标题，原文中的注解序号请包裹在[]号中
5. 只返回匹配的原文内容，不要添加任何解释或额外文字
"""


def query_api(prompt, key_manager, api_url, model, retries=MAX_RETRIES, system_prompt=None):
    """使用密钥池查询API"""
    for attempt in range(retries):
        api_key = key_manager.get_key()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt or get_system_prompt()},
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


def align_pair(args, key_manager, api_url, model, system_prompt):
    """对齐处理函数"""
    index, target_text, source_text = args
    try:
        prompt = f"译文:\n{target_text}\n\n原文:\n{source_text}"
        aligned_source = query_api(prompt, key_manager, api_url, model, system_prompt=system_prompt)

        if aligned_source == "API_ERROR":
            return (index, target_text, "API请求失败")
        elif "无对应内容" in aligned_source or not aligned_source:
            return (index, target_text, "无对应内容")
        return (index, target_text, aligned_source)
    except Exception as e:
        print(f"处理第{index}条出错: {str(e)}")
        return (index, target_text, "处理错误")


def debug_sample(df, key_manager, api_url, model, system_prompt, sample_count=DEBUG_SAMPLE):
    """调试模式"""
    print(f"\n=== 调试模式(前{sample_count}个样本) ===")
    samples = []

    # 自动检测列名
    target_col = 'target_text' if 'target_text' in df.columns else 'zh_text' if 'zh_text' in df.columns else 'cn_text'
    source_col = 'source_text' if 'source_text' in df.columns else 'en_text'

    for i, row in df.head(sample_count).iterrows():
        target_text = row[target_col]
        source_text = row[source_col]
        start_time = time.time()
        result = align_pair((i, target_text, source_text), key_manager, api_url, model, system_prompt)
        elapsed = time.time() - start_time

        samples.append({
            "译文": result[1],
            "对齐结果": result[2],
            "耗时(秒)": round(elapsed, 2)
        })
        print(f"\n样本{i+1} (耗时:{elapsed:.2f}s):")
        print(f"译文: {result[1][:100]}..." if len(result[1]) > 100 else f"译文: {result[1]}")
        print(f"结果: {result[2][:100]}..." if len(result[2]) > 100 else f"结果: {result[2]}")
        time.sleep(0.5)

    pd.DataFrame(samples).to_csv('debug_sample.csv', index=False, encoding='utf-8-sig')
    print("\n调试结果已保存到 debug_sample.csv")


def process_full(df, key_manager, api_url, model, system_prompt, output_csv,
                 checkpoint_file, thread_count, resume=True):
    """完整处理模式（支持断点续传）"""

    # 初始化检查点管理器
    checkpoint = CheckpointManager(checkpoint_file, output_csv)

    # 尝试恢复进度
    if resume:
        checkpoint.load()

    # 自动检测列名
    target_col = 'target_text' if 'target_text' in df.columns else 'zh_text' if 'zh_text' in df.columns else 'cn_text'
    source_col = 'source_text' if 'source_text' in df.columns else 'en_text'

    # 获取待处理的索引
    pending_indices = checkpoint.get_pending_indices(len(df))

    if not pending_indices:
        print("所有任务已完成！")
        return

    completed_before = checkpoint.get_results_count()
    print(f"总任务: {len(df)}, 已完成: {completed_before}, 待处理: {len(pending_indices)}")

    failed_count = 0

    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        # 只提交待处理的任务
        tasks = [(i, df.iloc[i][target_col], df.iloc[i][source_col])
                 for i in pending_indices]

        futures = {
            executor.submit(align_pair, task, key_manager, api_url, model, system_prompt): task[0]
            for task in tasks
        }

        try:
            for future in tqdm(as_completed(futures), total=len(tasks),
                              initial=0, desc="处理进度"):
                idx = futures[future]
                try:
                    _, target, source = future.result()
                    checkpoint.mark_completed(idx, target, source)
                except Exception as e:
                    print(f"任务{idx}异常: {str(e)}")
                    failed_count += 1
        except KeyboardInterrupt:
            print("\n\n检测到中断信号，正在保存进度...")
            checkpoint.save_final()
            completed_now = checkpoint.get_results_count()
            print(f"进度已保存！本次完成: {completed_now - completed_before} 条")
            print(f"下次运行将自动从第 {completed_now} 条继续")
            raise

    # 保存最终结果
    checkpoint.save_final()

    final_count = checkpoint.get_results_count()
    print(f"\n处理完成！")
    print(f"  总计: {final_count} 条")
    print(f"  本次新增: {final_count - completed_before} 条")
    print(f"  失败: {failed_count} 条")
    print(f"结果已保存到 {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='AI文本对齐工具 - 使用大语言模型进行精确对齐（支持断点续传）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python ai_aligned_text.py -i aligned_texts.csv -o result.csv
  python ai_aligned_text.py --source-lang English --target-lang Chinese
  python ai_aligned_text.py --threads 20 --debug-only
  python ai_aligned_text.py --no-resume  # 不恢复进度，从头开始

API密钥配置方式（按优先级）：
  1. 环境变量: export ALIYUN_DEEPSEEK_API_KEYS="key1,key2,key3"
  2. 配置文件: api_keys.txt（每行一个密钥）
  3. 配置文件: config.json（{"api_keys": ["key1", "key2"]}）

断点续传：
  - 程序会自动保存进度到 checkpoint.json
  - 中断后重新运行会自动恢复
  - 使用 --no-resume 可以忽略已有进度从头开始
        '''
    )

    parser.add_argument('-i', '--input',
                        default=DEFAULT_INPUT_CSV,
                        help=f'输入CSV文件路径（默认: {DEFAULT_INPUT_CSV}）')
    parser.add_argument('-o', '--output',
                        default=DEFAULT_OUTPUT_CSV,
                        help=f'输出CSV文件路径（默认: {DEFAULT_OUTPUT_CSV}）')
    parser.add_argument('--checkpoint',
                        default=DEFAULT_CHECKPOINT_FILE,
                        help=f'检查点文件路径（默认: {DEFAULT_CHECKPOINT_FILE}）')
    parser.add_argument('--source-lang',
                        default=None,
                        help='源语言名称（如 "English"、"日文"）')
    parser.add_argument('--target-lang',
                        default=None,
                        help='目标语言名称（如 "Chinese"、"中文"）')
    parser.add_argument('--api-url',
                        default=DEFAULT_API_URL,
                        help=f'API URL（默认: {DEFAULT_API_URL}）')
    parser.add_argument('--model',
                        default=DEFAULT_MODEL,
                        help=f'模型名称（默认: {DEFAULT_MODEL}）')
    parser.add_argument('--threads',
                        type=int,
                        default=DEFAULT_THREAD_COUNT,
                        help=f'线程数（默认: {DEFAULT_THREAD_COUNT}）')
    parser.add_argument('--debug-samples',
                        type=int,
                        default=DEBUG_SAMPLE,
                        help=f'调试样本数（默认: {DEBUG_SAMPLE}）')
    parser.add_argument('--debug-only',
                        action='store_true',
                        help='仅运行调试模式，不处理完整文件')
    parser.add_argument('--skip-debug',
                        action='store_true',
                        help='跳过调试模式，直接处理完整文件')
    parser.add_argument('--no-resume',
                        action='store_true',
                        help='不恢复进度，从头开始处理')

    args = parser.parse_args()

    # 加载API密钥
    api_keys = load_api_keys()
    if not api_keys:
        print("错误: 未找到API密钥！")
        print("请通过以下方式之一配置API密钥：")
        print("  1. 环境变量: export ALIYUN_DEEPSEEK_API_KEYS='key1,key2,key3'")
        print("  2. 创建文件: api_keys.txt（每行一个密钥）")
        print("  3. 创建文件: config.json（{\"api_keys\": [\"key1\", \"key2\"]}）")
        return 1

    print(f"已加载 {len(api_keys)} 个API密钥")
    print(f"线程数: {args.threads}")

    key_manager = KeyManager(api_keys)
    system_prompt = get_system_prompt(args.source_lang, args.target_lang)

    # 读取输入文件
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"错误: 文件未找到 - {args.input}")
        return 1

    print(f"已加载 {len(df)} 条记录")

    # 调试模式
    if not args.skip_debug:
        debug_sample(df, key_manager, args.api_url, args.model, system_prompt, args.debug_samples)
        if args.debug_only:
            return 0

        # 确认继续
        response = input("\n是否继续处理完整文件？(y/n): ")
        if response.lower() != 'y':
            print("已取消处理")
            return 0

    # 完整处理
    try:
        process_full(df, key_manager, args.api_url, args.model, system_prompt,
                     args.output, args.checkpoint, args.threads,
                     resume=not args.no_resume)
    except KeyboardInterrupt:
        print("\n程序已中断，进度已保存")
        return 0

    return 0


if __name__ == "__main__":
    exit(main())
