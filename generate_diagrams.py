#!/usr/bin/env python3
"""生成对齐原理说明文档的配图"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_gale_church_diagram():
    """创建Gale-Church算法原理图"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # 标题
    ax.text(5, 5.6, 'Gale-Church 算法原理', fontsize=16, fontweight='bold', ha='center')
    ax.text(5, 5.2, '基于句子长度的统计规律进行对齐', fontsize=11, ha='center', color='#666')

    # 左侧：英文句子
    ax.text(1.5, 4.5, '英文句子', fontsize=12, fontweight='bold', ha='center')

    # 英文句子框
    en_boxes = [
        (0.3, 3.8, 2.4, 0.5, '120 chars'),
        (0.3, 3.0, 1.6, 0.5, '80 chars'),
        (0.3, 2.2, 0.8, 0.5, '40'),
        (0.3, 1.4, 1.0, 0.5, '50'),
    ]

    for x, y, w, h, label in en_boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                               facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, fontsize=9, ha='center', va='center')

    # 右侧：中文句子
    ax.text(8.5, 4.5, '中文句子', fontsize=12, fontweight='bold', ha='center')

    zh_boxes = [
        (7.3, 3.8, 2.4, 0.5, '65字'),
        (7.3, 3.0, 2.0, 0.5, '45字'),
        (7.3, 1.8, 2.4, 0.9, '70字'),
    ]

    for x, y, w, h, label in zh_boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                               facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, fontsize=9, ha='center', va='center')

    # 箭头和比例说明
    ax.annotate('', xy=(7.1, 4.05), xytext=(2.9, 4.05),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2))
    ax.text(5, 4.2, '1:1  约1.8:1', fontsize=9, ha='center', color='#4CAF50')

    ax.annotate('', xy=(7.1, 3.25), xytext=(2.1, 3.25),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2))
    ax.text(5, 3.4, '1:1', fontsize=9, ha='center', color='#4CAF50')

    # 合并箭头
    ax.annotate('', xy=(3.5, 2.0), xytext=(1.3, 2.45),
                arrowprops=dict(arrowstyle='->', color='#9C27B0', lw=1.5))
    ax.annotate('', xy=(3.5, 2.0), xytext=(1.5, 1.65),
                arrowprops=dict(arrowstyle='->', color='#9C27B0', lw=1.5))

    ax.text(4.2, 2.0, '2:1', fontsize=9, ha='center', color='#9C27B0', fontweight='bold')
    ax.text(5, 1.6, '两句合并', fontsize=9, ha='center', color='#9C27B0')

    ax.annotate('', xy=(7.1, 2.25), xytext=(5.8, 2.0),
                arrowprops=dict(arrowstyle='->', color='#9C27B0', lw=1.5))

    # 底部说明
    ax.text(5, 0.6, '算法穷举各种对齐方式（1:1, 1:2, 2:1...）', fontsize=10, ha='center', color='#333')
    ax.text(5, 0.2, '选择"长度比例最符合统计规律"的组合', fontsize=10, ha='center', color='#333')

    plt.tight_layout()
    plt.savefig('images/gale_church.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print('已生成: images/gale_church.png')


def create_comparison_diagram():
    """创建传统方法 vs AI方法对比图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # 标题
    ax.text(6, 6.6, '传统方法 vs AI方法', fontsize=16, fontweight='bold', ha='center')

    # 左侧：传统方法
    rect1 = FancyBboxPatch((0.5, 3.5), 5, 2.5, boxstyle="round,pad=0.05",
                            facecolor='#FAFAFA', edgecolor='#9E9E9E', linewidth=2)
    ax.add_patch(rect1)
    ax.text(3, 5.7, '传统：基于统计', fontsize=13, fontweight='bold', ha='center', color='#616161')

    items_left = ['• 句子长度比例', '• 词典匹配', '• 锚点词对齐']
    for i, item in enumerate(items_left):
        ax.text(1, 5.0 - i*0.5, item, fontsize=11, va='center', color='#757575')

    # 左侧判断框
    rect_left = FancyBboxPatch((1, 1.8), 4, 1.2, boxstyle="round,pad=0.03",
                                facecolor='#E8EAF6', edgecolor='#3F51B5', linewidth=2)
    ax.add_patch(rect_left)
    ax.text(3, 2.5, '"120字符对65字"', fontsize=10, ha='center', color='#3F51B5')
    ax.text(3, 2.1, '比例合理，配对！', fontsize=10, ha='center', color='#3F51B5')

    ax.annotate('', xy=(3, 1.8), xytext=(3, 3.3),
                arrowprops=dict(arrowstyle='->', color='#3F51B5', lw=2))

    # 左侧问题
    ax.text(3, 1.3, '问题：', fontsize=11, fontweight='bold', ha='center', color='#C62828')
    problems = ['意译就傻眼', '断句错就全错', '需要大量返工']
    for i, p in enumerate(problems):
        ax.text(3, 0.9 - i*0.35, f'× {p}', fontsize=10, ha='center', color='#E57373')

    # 右侧：AI方法
    rect2 = FancyBboxPatch((6.5, 3.5), 5, 2.5, boxstyle="round,pad=0.05",
                            facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=2)
    ax.add_patch(rect2)
    ax.text(9, 5.7, 'AI：基于理解', fontsize=13, fontweight='bold', ha='center', color='#2E7D32')

    items_right = ['• 语义理解', '• 上下文分析', '• 跨语言知识']
    for i, item in enumerate(items_right):
        ax.text(7, 5.0 - i*0.5, item, fontsize=11, va='center', color='#4CAF50')

    # 右侧判断框
    rect_right = FancyBboxPatch((7, 1.8), 4, 1.2, boxstyle="round,pad=0.03",
                                 facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=2)
    ax.add_patch(rect_right)
    ax.text(9, 2.5, '"这两段都在讲', fontsize=10, ha='center', color='#2E7D32')
    ax.text(9, 2.1, '我已经老了..."', fontsize=10, ha='center', color='#2E7D32')

    ax.annotate('', xy=(9, 1.8), xytext=(9, 3.3),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2))

    # 右侧优势
    ax.text(9, 1.3, '优势：', fontsize=11, fontweight='bold', ha='center', color='#2E7D32')
    advantages = ['意译也能对上', '理解段落边界', '几乎不用返工']
    for i, a in enumerate(advantages):
        ax.text(9, 0.9 - i*0.35, f'✓ {a}', fontsize=10, ha='center', color='#66BB6A')

    # VS
    circle = plt.Circle((6, 4.75), 0.4, color='#FF9800', ec='white', linewidth=2)
    ax.add_patch(circle)
    ax.text(6, 4.75, 'VS', fontsize=12, fontweight='bold', ha='center', va='center', color='white')

    plt.tight_layout()
    plt.savefig('images/comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print('已生成: images/comparison.png')


def create_workflow_diagram():
    """创建两阶段流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 标题
    ax.text(6, 7.6, '两阶段对齐流程', fontsize=16, fontweight='bold', ha='center')

    # 第一阶段背景
    rect_stage1 = FancyBboxPatch((0.3, 4.2), 11.4, 3),
    rect_bg1 = FancyBboxPatch((0.3, 4.5), 11.4, 2.8, boxstyle="round,pad=0.05",
                               facecolor='#FFF8E1', edgecolor='#FFA000', linewidth=2, linestyle='--')
    ax.add_patch(rect_bg1)
    ax.text(6, 7.1, '第一阶段：位置法粗筛（快）', fontsize=12, fontweight='bold', ha='center', color='#F57C00')

    # 文件1
    rect_f1 = FancyBboxPatch((0.8, 5.5), 2, 1),
    rect_file1 = FancyBboxPatch((0.8, 5.3), 2, 1.2, boxstyle="round,pad=0.02",
                                 facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
    ax.add_patch(rect_file1)
    ax.text(1.8, 6.1, '文件1', fontsize=11, fontweight='bold', ha='center', color='#1976D2')
    ax.text(1.8, 5.7, '第1段 0~15%', fontsize=9, ha='center', color='#1976D2')

    # 箭头
    ax.annotate('', xy=(4.3, 5.9), xytext=(3, 5.9),
                arrowprops=dict(arrowstyle='->', color='#FF9800', lw=2))
    ax.text(3.65, 6.15, '按位置', fontsize=9, ha='center', color='#FF9800')

    # 文件2
    rect_file2 = FancyBboxPatch((4.5, 5.3), 3.5, 1.2, boxstyle="round,pad=0.02",
                                 facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2)
    ax.add_patch(rect_file2)
    ax.text(6.25, 6.1, '文件2', fontsize=11, fontweight='bold', ha='center', color='#F57C00')

    # 高亮区域
    rect_highlight = FancyBboxPatch((4.7, 5.5), 1.2, 0.7, boxstyle="round,pad=0.01",
                                     facecolor='#FFCC80', edgecolor='#EF6C00', linewidth=2)
    ax.add_patch(rect_highlight)
    ax.text(7.3, 5.7, '取0~20%区间', fontsize=9, ha='center', color='#E65100')

    # 输出配对
    ax.annotate('', xy=(9.5, 5.9), xytext=(8.2, 5.9),
                arrowprops=dict(arrowstyle='->', color='#FF9800', lw=2))

    rect_pair = FancyBboxPatch((9.7, 5.3), 1.8, 1.2, boxstyle="round,pad=0.02",
                                facecolor='#FFF8E1', edgecolor='#FFA000', linewidth=2)
    ax.add_patch(rect_pair)
    ax.text(10.6, 5.9, '粗配对', fontsize=10, ha='center', color='#F57C00')

    # 第二阶段背景
    rect_bg2 = FancyBboxPatch((0.3, 0.8), 11.4, 3.4, boxstyle="round,pad=0.05",
                               facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=2, linestyle='--')
    ax.add_patch(rect_bg2)
    ax.text(6, 4.0, '第二阶段：AI精确匹配（准）', fontsize=12, fontweight='bold', ha='center', color='#2E7D32')

    # AI输入
    rect_ai_input = FancyBboxPatch((0.8, 1.5), 5, 2.2, boxstyle="round,pad=0.03",
                                    facecolor='white', edgecolor='#4CAF50', linewidth=2)
    ax.add_patch(rect_ai_input)
    ax.text(3.3, 3.4, 'Prompt:', fontsize=10, fontweight='bold', ha='center', color='#2E7D32')
    ax.text(1.1, 3.0, '文本A: 我已经老了，有一天...', fontsize=9, va='center', color='#333')
    ax.text(1.1, 2.5, "文本B: Un jour, j'étais âgée déjà,", fontsize=9, va='center', color='#333')
    ax.text(1.1, 2.15, '        dans le hall d\'un lieu public...', fontsize=9, va='center', color='#666')
    ax.text(1.1, 1.7, '请找出与文本A对应的段落', fontsize=9, va='center', color='#2E7D32', style='italic')

    # AI处理箭头
    ax.annotate('', xy=(7.3, 2.6), xytext=(6, 2.6),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2.5))

    # AI图标
    circle = plt.Circle((6.65, 2.6), 0.35, color='#4CAF50', ec='white', linewidth=2)
    ax.add_patch(circle)
    ax.text(6.65, 2.6, 'AI', fontsize=11, fontweight='bold', ha='center', va='center', color='white')

    # AI输出
    rect_ai_output = FancyBboxPatch((7.5, 1.5), 4, 2.2, boxstyle="round,pad=0.03",
                                     facecolor='#C8E6C9', edgecolor='#4CAF50', linewidth=2)
    ax.add_patch(rect_ai_output)
    ax.text(9.5, 3.4, '返回结果:', fontsize=10, fontweight='bold', ha='center', color='#2E7D32')
    ax.text(7.7, 2.9, "Un jour, j'étais âgée déjà,", fontsize=9, va='center', color='#1B5E20')
    ax.text(7.7, 2.5, 'dans le hall d\'un lieu public...', fontsize=9, va='center', color='#1B5E20')
    ax.text(7.7, 1.9, '（精确匹配，去掉多余内容）', fontsize=8, va='center', color='#66BB6A', style='italic')

    # 从第一阶段到第二阶段的箭头
    ax.annotate('', xy=(3.3, 4.2), xytext=(10.6, 5.1),
                arrowprops=dict(arrowstyle='->', color='#9E9E9E', lw=1.5,
                               connectionstyle="arc3,rad=0.3"))

    plt.tight_layout()
    plt.savefig('images/workflow.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print('已生成: images/workflow.png')


if __name__ == '__main__':
    import os
    os.makedirs('images', exist_ok=True)

    create_gale_church_diagram()
    create_comparison_diagram()
    create_workflow_diagram()
    print('\n所有图片已生成完毕！')
