import re
import argparse
import pandas as pd

# 常量设置
DEFAULT_MIN_LINE_LENGTH = 200  # 默认最小行长度
DEFAULT_WINDOW_SIZE = 0.01  # 冗余量


def preprocess_text(text, language=None):
    """
    通用文本预处理
    处理规则：
    1. 合并被错误分割的句子
    2. 保留章节标题（全大写单词单独成行，适用于拉丁字母语言）
    3. 保留真正的段落分隔
    4. 合并多余空格和空行

    Args:
        text: 输入文本
        language: 语言代码（可选），如 'en', 'zh', 'ja' 等
    """
    # 保留原始段落结构（先按双换行分割）
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    processed_paragraphs = []
    for para in paragraphs:
        # 按行处理
        lines = [line.strip() for line in para.split('\n') if line.strip()]
        merged_lines = []
        buffer = ""

        for i, line in enumerate(lines):
            # 章节标题判断（全大写且长度适中，适用于拉丁字母语言）
            is_heading = line.isupper() and 2 <= len(line.split()) <= 5

            # 判断是否应该合并的条件
            should_merge = (
                not is_heading
                and not re.search(r'[.!?。！？][""]?\s*\d*$', buffer)  # 支持中英文结束标点
                and not (buffer.endswith('"') and len(line) > 0 and line[0].isupper())
                and not (buffer and buffer[-1].isupper() and len(line) > 0 and line[0].isupper())
            )

            if should_merge:
                # 智能添加空格（前一行结尾不是连字符则加空格）
                # 对于中日韩等语言不需要空格
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

        # 后处理：确保引号和注释数字正确
        refined_para = []
        for line in merged_lines:
            # 处理引号后跟数字注释的情况（如 "Hammer" 1）
            line = re.sub(r'([""])\s*(\d+)', r'\1\2', line)
            refined_para.append(line)

        processed_paragraphs.append('\n'.join(refined_para))

    # 最终合并（保留段落间单空行）
    return '\n\n'.join(processed_paragraphs)


def _is_cjk_text(text):
    """判断文本是否主要是中日韩文字"""
    if not text:
        return False
    cjk_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff' or
                    '\u3040' <= char <= '\u309f' or  # 日文平假名
                    '\u30a0' <= char <= '\u30ff' or  # 日文片假名
                    '\uac00' <= char <= '\ud7af')    # 韩文
    return cjk_count > len(text) * 0.3


def merge_short_paragraphs(text, min_length=DEFAULT_MIN_LINE_LENGTH):
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
        ranges.append((
            para,
            start/total_len,
            end/total_len
        ))

    return ranges


def align_texts(source_text, target_text, min_length=DEFAULT_MIN_LINE_LENGTH,
                window_size=DEFAULT_WINDOW_SIZE):
    """
    主对齐函数

    Args:
        source_text: 源语言文本（原文）
        target_text: 目标语言文本（译文，用于确定分段）
        min_length: 最小段落长度
        window_size: 窗口冗余量
    """
    # 预处理源文本
    source_text = preprocess_text(source_text)

    # 处理目标文本：先清理再合并
    target_cleaned = '\n'.join([line.strip() for line in target_text.split('\n') if line.strip()])
    target_paragraphs = merge_short_paragraphs(target_cleaned, min_length)
    target_ranges = calculate_ranges(target_cleaned, target_paragraphs)

    # 准备结果
    results = []
    for target, target_start, target_end in target_ranges:
        # 计算源文本对应范围（带冗余窗口）
        source_start = max(0, int((target_start - window_size*2) * len(source_text)))
        source_end = min(len(source_text), int((target_end + window_size) * len(source_text)))

        results.append({
            'source_text': source_text[source_start:source_end],
            'target_text': target,
            'source_start_pct': target_start - window_size*2,
            'source_end_pct': target_end + window_size,
            'target_start_pct': target_start,
            'target_end_pct': target_end,
            'target_line_length': len(target)
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='文本对齐工具 - 适用于任意语言对',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python aligned_texts.py -s english.txt -t chinese.txt -o output.csv
  python aligned_texts.py --source source.txt --target target.txt --min-length 150
        '''
    )

    parser.add_argument('-s', '--source',
                        default='english.txt',
                        help='源语言文件路径（原文）')
    parser.add_argument('-t', '--target',
                        default='chinese.txt',
                        help='目标语言文件路径（译文）')
    parser.add_argument('-o', '--output',
                        default='aligned_texts.csv',
                        help='输出CSV文件路径')
    parser.add_argument('--min-length',
                        type=int,
                        default=DEFAULT_MIN_LINE_LENGTH,
                        help=f'最小段落长度（默认: {DEFAULT_MIN_LINE_LENGTH}）')
    parser.add_argument('--window-size',
                        type=float,
                        default=DEFAULT_WINDOW_SIZE,
                        help=f'窗口冗余量（默认: {DEFAULT_WINDOW_SIZE}）')
    parser.add_argument('--encoding',
                        default='utf-8',
                        help='文件编码（默认: utf-8）')

    args = parser.parse_args()

    # 读取文件
    try:
        with open(args.source, 'r', encoding=args.encoding) as f:
            source_text = f.read()
        with open(args.target, 'r', encoding=args.encoding) as f:
            target_text = f.read()
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e.filename}")
        return 1
    except UnicodeDecodeError:
        print(f"错误: 文件编码错误，请尝试指定 --encoding 参数")
        return 1

    # 执行对齐
    df = align_texts(source_text, target_text, args.min_length, args.window_size)

    # 保存结果
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"对齐完成！共 {len(df)} 个段落")
    print(f"结果已保存到: {args.output}")
    print(f"参数: 最小长度={args.min_length}, 窗口={args.window_size}")

    return 0


if __name__ == "__main__":
    exit(main())
