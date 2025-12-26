# 双语文本对齐工具

基于大语言模型的双语文本对齐工具，适用于翻译校对、平行语料库构建等场景。

## 特点

- **两阶段对齐**：位置法粗筛 + AI精确匹配
- **多语言支持**：自动识别语言，无需手动指定
- **断点续传**：支持中断恢复，大文件处理无压力
- **多线程并行**：支持多API密钥轮换，提高处理速度

## 使用方法

```bash
# 输入两个文件，输出对齐好的CSV
python align.py file1.txt file2.txt

# 指定输出文件
python align.py file1.txt file2.txt -o result.csv

# 仅做粗对齐（不调用AI）
python align.py file1.txt file2.txt --skip-ai
```

## 配置

1. 复制 `api_keys.txt.example` 为 `api_keys.txt`
2. 填入你的API密钥（每行一个）

或通过环境变量配置：
```bash
export ALIYUN_DEEPSEEK_API_KEYS="key1,key2,key3"
```

## 输出格式

| _index | text_a | text_b |
|--------|--------|--------|
| 0 | 第一个文件的段落 | 第二个文件对应的段落 |
| 1 | ... | ... |

## 原理说明

详见 [对齐原理说明.md](对齐原理说明.md)

![工作流程](images/workflow.png)

## 依赖

```bash
pip install pandas requests tqdm matplotlib
```

## License

MIT
