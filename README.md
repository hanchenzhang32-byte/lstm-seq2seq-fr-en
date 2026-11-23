# LSTM 法→英短句机器翻译（Seq2Seq，无 Attention）

French→English short-sentence translation with a PyTorch LSTM encoder–decoder (no attention), trained on 8,000 Tatoeba sentence pairs.

## 结果

| 项目 | 值 |
|---|---|
| 最佳 checkpoint | epoch 11（验证 loss 2.9008，按验证集选择，提前停止于 epoch 14） |
| 测试集 corpus BLEU | **11.39**（1,000 句，贪心解码，无 Teacher Forcing） |
| sacreBLEU signature | `nrefs:1\|case:mixed\|eff:no\|tok:none\|smooth:exp\|version:2.6.0` |

![loss 曲线](outputs/loss_curve.png)

完整实验报告（数据处理、模型、10 条测试样例、失败案例分析、问答）见 [report.md](report.md)。

## 运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

下载数据（ManyThings.org 会拒绝不带浏览器 User-Agent 的请求，也可以用浏览器手动下载后放进 `data/`）：

```bash
mkdir -p data
curl -L -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36" -o data/fra-eng.zip https://www.manythings.org/anki/fra-eng.zip
unzip -o data/fra-eng.zip -d data
```

```bash
python check_lesson1.py   # 数据处理自检（12/12）
python train.py           # 训练，CPU 约 40 秒 → outputs/best_model.pt、loss_curve.png、history.json
python evaluate.py        # 测试 BLEU、10 条样例、失败案例候选 → outputs/test_*.json*
```

注：数据文件会随 ManyThings.org 更新而变化。本实验使用的是 2026-02-13 版本，共 240,521 行；换成其他版本后，句对数量和结果都会有差异。

## 数据来源

- Tatoeba 法语—英语平行语料，由 ManyThings.org 整理：https://www.manythings.org/anki/
- 许可：CC-BY 2.0 (France)，署名：www.manythings.org/anki 与 tatoeba.org
- 本仓库不包含原始数据文件；处理时第三列的逐句署名信息不参与训练

## 文件

| 文件 | 内容 |
|---|---|
| `data_prep.py` | 规范化、分词、过滤、去重、打乱、划分 |
| `dataset.py` | 词表、Dataset、DataLoader、批内补齐、目标序列错位 |
| `model.py` | Encoder / Decoder / Seq2Seq，贪心解码 |
| `train.py` | 训练循环、忽略 PAD 的 loss、梯度裁剪、按验证 loss 保存最佳 checkpoint |
| `evaluate.py` | `translate()`、sacreBLEU、样例与失败案例 |
| `report.md` | 实验报告与问答 |
| `outputs/` | 最佳模型、loss 曲线、训练与评估日志、全部测试译文 |
| `basics/` | 学习用的小练习：文字转数字、手写 RNN / LSTM 单步 |

## 预处理规则

1. Unicode NFC 规范化；不换行空格、窄空格等统一为普通空格，零宽字符删除；弯引号 ’ ‘ 转为 '
2. 全部小写
3. 分词正则 `\w+|[^\w\s]`：连续字母/数字为一个 token，每个标点单独为一个 token
