# 实验报告：基于 PyTorch 与 LSTM 的法→英短句机器翻译

## 1. 数据

**来源**：Tatoeba 法语—英语平行语料，由 ManyThings.org 整理（https://www.manythings.org/anki/ ，下载 `fra-eng.zip`，文件日期 2026-02-13）。许可 CC-BY 2.0 (France)，署名 www.manythings.org/anki 与 tatoeba.org。`data/fra.txt` 原样保留，第三列逐句署名信息未删改，仅不参与训练。

**预处理**（作业原文规则列表缺第 1 条，本项目自行规定并统一用于训练、推理和 BLEU 参考文本）：

1. Unicode NFC 规范化；不换行空格 `\xa0`、窄不换行空格 ` `、细空格 ` ` 统一为普通空格，零宽空格与软连字符删除；弯引号 `’ ‘` 转为 `'`
2. 全部小写
3. 分词正则 `\w+|[^\w\s]`：连续字母/数字为一个 token，每个标点单独为一个 token（如 `don't` → `don ' t`）

**筛选与划分**：

| 步骤 | 数量 |
|---|---|
| 原始句对 | 240,521 |
| 去掉任一侧为空或超过 10 个 token（标点计入） | −68,258 |
| 按处理后法语去重，保留原文件中第一条合格译文 | −22,849 |
| 合格句对 | 149,414 |
| `random.Random(42).shuffle` 后取前 10,000 | 训练 8,000 / 验证 1,000 / 测试 1,000 |

**词表**（只用训练集建立，含 `<PAD> <SOS> <EOS> <UNK>`）：法语 5,778，英语 4,088。验证集、测试集法语 token 中未登录词比例分别为 5.9% 和 5.7%，测试集有 352/1000 句至少含一个未登录词。

**批次**：`Dataset` 存编号序列，`collate_fn` 在批内补齐并返回法语真实长度。目标序列错位一位：

```
解码器输入：<SOS> i am happy .
预测目标：  i am happy . <EOS>
```

## 2. 模型

不带 Attention 的 Encoder–Decoder，共 2,054,392 个参数（`model.py`）：

- **Encoder**：法语编号 → `nn.Embedding(5778, 128)` → 单层单向 `nn.LSTM(128, 128)`。用 `pack_padded_sequence` 按真实长度打包，最终 `(h, c)` 取自每句最后一个真实词，而不是读完 `<PAD>` 之后的状态（已验证：同一句单独编码与在补齐批次中编码，得到的 h 完全相同）。
- **状态传递**：编码器最终的 `h` 和 `c` 一起作为解码器 LSTM 的初始状态。
- **Decoder**：上一个英语 token → `nn.Embedding(4088, 128)`（独立的嵌入层）→ 单层单向 `nn.LSTM(128, 128)` → `nn.Linear(128, 4088)`，输出原始 logits。

## 3. 训练

| 超参数 | 值 |
|---|---|
| 词嵌入 / 隐藏维度 | 128 / 128 |
| LSTM 层数 | 1 |
| Batch size | 32 |
| 优化器 / 学习率 | Adam / 0.001 |
| 梯度裁剪 | 最大范数 1.0 |
| Teacher Forcing 比例 | 1.0 |
| 最大轮数 / 提前停止 | 20 / 验证 loss 连续 3 轮不下降 |
| 随机种子 | 42（`random`、`torch`、DataLoader 打乱均固定） |

- 损失：`CrossEntropyLoss(ignore_index=pad_id)`，输入原始 logits。
- 每轮 loss 按非 PAD token 数平均：每批 loss 乘以该批非 PAD token 数后累加，再除以总 token 数。
- **验证 loss 同样使用 Teacher Forcing**（解码器输入真实的上一个词），因此它衡量的是"给定正确前文时预测下一个词"的能力，与独立生成译文不同。
- 只根据验证 loss 选择最佳 checkpoint；训练脚本中测试集完全不参与。
- 在 CPU 上训练，每轮约 3 秒。以相同种子重跑一次，逐轮 loss 完全一致。

![loss 曲线](outputs/loss_curve.png)

| epoch | train loss | val loss |
|---|---|---|
| 1 | 4.6747 | 3.9269 |
| 2 | 3.5785 | 3.5291 |
| 3 | 3.1878 | 3.3312 |
| 4 | 2.8986 | 3.1847 |
| 5 | 2.6560 | 3.0929 |
| 6 | 2.4417 | 3.0179 |
| 7 | 2.2435 | 2.9630 |
| 8 | 2.0600 | 2.9335 |
| 9 | 1.8898 | 2.9036 |
| 10 | 1.7305 | 2.9012 |
| **11** | **1.5802** | **2.9008（最佳）** |
| 12 | 1.4363 | 2.9032 |
| 13 | 1.3028 | 2.9115 |
| 14 | 1.1798 | 2.9267（停止） |

**最佳 checkpoint：epoch 11**，验证 loss 2.9008。第 9 轮之后验证 loss 基本持平，而训练 loss 继续快速下降，两者差距越拉越大，说明模型开始过拟合 8,000 句训练数据。

## 4. 测试集 BLEU

使用最佳 checkpoint，在完整测试集 1,000 句上用贪心解码独立生成译文（不使用 Teacher Forcing，不输入参考译文），每句最多 15 个 token。预测 token 与参考 token 都用空格连接，参考文本为原始预处理结果，未把未登录词替换成 `<UNK>`。

| 项目 | 值 |
|---|---|
| **corpus BLEU** | **11.39** |
| 详细 | `BLEU = 11.39 45.8/14.7/7.7/3.3 (BP = 1.000 ratio = 1.050 hyp_len = 7036 ref_len = 6700)` |
| 软件版本 | sacreBLEU 2.6.0 |
| signature | `nrefs:1\|case:mixed\|eff:no\|tok:none\|smooth:exp\|version:2.6.0` |

调用方式为 `BLEU(tokenize="none", force=True).corpus_score(predictions, [references])`。`force=True` 只是关闭 sacreBLEU 对"文本看起来已分词"的提示（本作业要求的就是分词后评分），不影响分数。

**分组 BLEU**（辅助分析）：

| 分组 | 句数 | BLEU |
|---|---|---|
| 法语 1–5 个 token | 176 | 17.45 |
| 法语 6–7 个 token | 343 | 10.76 |
| 法语 8 个 token | 172 | 10.77 |
| 法语 9–10 个 token | 309 | 10.28 |
| 源句含未登录词 | 352 | 8.73 |
| 源句无未登录词 | 648 | 12.68 |

1-gram 精确率 45.8%，说明模型大多能选对相关的英语词；4-gram 精确率只有 3.3%，说明能连续说对的长片段很少。与参考译文完全一致的有 3/1000 句。

## 5. 10 条测试样例

用 `random.Random(42).sample(range(1000), 10)` 抽取（测试集下标 25, 104, 114, 142, 228, 250, 281, 654, 754, 759）：

| # | 法语输入 | 参考英文 | 模型输出 |
|---|---|---|---|
| 1 | tu ne sais rien . | you know nothing . | you can ' t be . |
| 2 | personne ne te demande de manger ça . | no one is asking you to eat that . | no one would you like to do that . |
| 3 | vous ne pouvez pas avoir trente ans ! | you can ' t be thirty years old . | you can ' t have a little . |
| 4 | ceux - ci sont véritables . | these are genuine . | they are singing . |
| 5 | y a - t - il un message ? | is there a message ? | is it a few minutes ? |
| 6 | c ' est un charmant jeune homme . | he ' s a lovely young man . | it ' s a man in the woods . |
| 7 | j ' aurais aimé avoir meilleure mémoire . | i wish i had a better memory . | i can ' t wait for you . |
| 8 | je suis à l ' extérieur . | i ' m outside . | i ' m in the room . |
| 9 | et si je dis non ? | what if i say no ? | what i can ' t you ? |
| 10 | il a décidé de ne pas y aller . | he decided not to go . | he is no longer yet . |

这 10 条比较典型：模型通常能抓住**句子框架**，比如主语（no one / he / i ' m）、否定结构（can ' t）、疑问句和句末标点，但**实义词**常常被换成训练集里的高频搭配（in the room、a few minutes、a little）。其中第 2、3、6、8 条前半句基本正确，后半句跑偏。

## 6. 失败案例分析

### 案例 1：重复生成

| 法语 | 参考 | 输出 |
|---|---|---|
| c ' est vraiment beau . | it ' s really beautiful . | it ' s really **really** good . |
| tu as toujours le choix . | you always have a choice . | you ' ve always **always** right . |
| je viens juste de rentrer du travail . | i just got back from work . | i just **just** wanted to the same . |

测试集中有 17 句出现相邻 token 重复。**原因**：解码器"已经说过什么"的信息只能隐含在 128 维的 `h, c` 里，没有显式记录。生成 `really` 之后，状态变化不大，下一步 `really` 仍然是概率最大的词。贪心解码每步只取最大值、不能回头修改，一旦重复就停不下来。另外，训练时解码器输入的永远是正确的前文，从没见过"自己刚说错"的情况，这正是 Teacher Forcing 带来的曝光偏差。

### 案例 2：未知词

| 法语（未登录词加粗） | 参考 | 输出 |
|---|---|---|
| **corrigez** les mots **soulignés** . | correct the underlined words . | thanks down the rules . |
| le **tissu** était **teinté** d ' un rouge **éclatant** . | the cloth was dyed bright red . | the meeting was canceled of her dreams . |
| je ne me **sentais** pas bien . | i didn ' t feel well . | i don ' t know who . |

**原因**：含未登录词的句子 BLEU 只有 8.73，无未登录词的是 12.68。词表只用训练集建立，并且保留了所有训练词（`min_freq=1`），所以**训练时法语输入里从没出现过 `<UNK>`，它的词向量一直停留在随机初始值**。测试时多个关键词都变成同一个未经训练的向量，编码器拿不到任何语义。第 3 例中 `sentais` 只是 `sentir`（感觉）的未完成过去时变位，训练集里可能有它的其他变位，但按词切分时就完全是"新词"。同样的原因也让模型**从不输出 `<UNK>`**：英语训练目标里没有 `<UNK>`，模型从没学过生成它。

### 案例 3：漏译

| 法语 | 参考 | 输出 |
|---|---|---|
| sa maison est au pied de la montagne . | her house is at the foot of the mountain . | his house is in the room . |
| reposons - nous . | let ' s take a rest . | we need . |
| nous vous aiderons à faire cela . | we ' ll help you do that . | we should do you . |

**原因**：解码器在主语或开头几个词之后就失去了源句信息，用一个常见短语或 `<EOS>` 草草收尾，`au pied de la montagne` 整段没有被翻译。整句法语只被压缩进一个 128 维的 `(h, c)`，解码越往后，这个初始状态的影响越弱，模型越依赖英语自身的语言模型习惯（例如 `is in the room` 在训练集中很常见）。第 2 例同时有未登录词 `reposons`，第 3 例的 `aiderons`（将来时）也是未登录词，与案例 2 叠加。

### 案例 4：较长句子语义完全漂移（不含未登录词）

| 法语 | 参考 | 输出 |
|---|---|---|
| la fête s ' est terminée à neuf heures . | the party ended at nine . | food is getting and all of them . |
| prends un peu de sommeil , d ' accord ? | get some sleep , okay ? | are not going to get my friend . |

**原因**：这两句没有未登录词，仍然完全翻错。法语 9–10 个 token 的句子 BLEU 是 10.28，1–5 个 token 的是 17.45。没有 Attention 时，解码器每一步只能看到同一个固定长度的句子向量，不能回头查看源句的某个具体词。句子越长，信息压缩越严重，`s ' est terminée`（结束了）这样由多个 token 组成的结构就更难保留下来。

### 补充：BLEU 低但译文并非全错

| 法语 | 参考 | 输出 | 说明 |
|---|---|---|---|
| nous y allons maintenant . | we ' re going now . | we ' re going to go now . | 语义基本正确，多了 `to go` |
| je dois entrer . | i need to go in . | i have to go . | `have to` 与 `need to` 同义，只漏了 `in` |
| sa maison est … | **her** house … | **his** house … | 法语 `sa` 的阴阳性跟随宾语 `maison`，不反映主人性别，his 和 her 都可能正确 |

参考译文只有一个，不是唯一正确的表达。BLEU 只按 n-gram 重合计分，会把同义改写和合理歧义都判为错误。

## 7. 问题回答

**为什么编码器和解码器可以处理长度不同的输入与输出？**
LSTM 在每个时间步重复使用同一组权重，循环多少步由序列本身决定，所以任意长度的输入都能读完，并得到大小固定的 `(h, c)`。编码器读完整句法语后，把它压缩成这个固定大小的状态。解码器从这个状态出发，每步生成一个英语词，直到生成 `<EOS>`（或达到 15 个 token 上限）才停止，所以输出长度由模型自己决定，与输入长度无关。批量训练时，长度不同的句子通过 `<PAD>` 补齐，再配合真实长度（`pack_padded_sequence`）和损失中的 `ignore_index`，让补齐部分既不影响编码结果，也不影响梯度。

**LSTM 的 h 和 c 分别承担什么作用？**
`c`（记忆状态）是一条贯穿各时间步的"传送带"。每一步只通过遗忘门按比例保留、通过输入门加入新信息，主要是逐元素的加法和乘法，因此梯度沿 `c` 回传时不容易消失，能保存较长期的信息。`h`（隐藏状态）是 `c` 经过 tanh 和输出门筛选后"本步对外展示"的部分：它是当前时间步的输出（解码器中再经过 Linear 得到 logits），同时也作为下一步计算各个门的输入。在本模型中，编码器最终的 `h` 和 `c` 一起交给解码器：`c` 携带整句的长期记忆，`h` 决定解码器第一步的门控和输出。

**Teacher Forcing 与实际推理有什么不同？可能造成什么问题？**
训练时（比例 1.0），解码器每一步的输入是**真实的**上一个英语词。推理时没有参考译文，每一步的输入是**模型自己上一步的预测**。这会造成**曝光偏差**：训练中模型从没见过"前文有错"的情况，推理时一旦某一步预测错，后续输入就偏离了训练分布，错误会逐步累积，表现为后半句跑题、重复生成或漏译（见第 6 节）。同理，用 Teacher Forcing 算出的验证 loss（2.90）只反映"前文正确时预测下一个词"的能力，比实际生成的质量乐观，所以必须另外用无 Teacher Forcing 的生成结果计算 BLEU。

**为什么计算损失时应忽略 PAD？为什么测试集不能参与词表构建？**
PAD 只是为了让批次成为矩形张量而补的占位符，不是要预测的内容。如果计入损失：(1) 模型会花力气学习"在句末之后预测 `<PAD>`"这种毫无意义的任务，因为它非常容易学，还会拉低 loss，让 loss 看起来比实际好；(2) 每句的权重会随所在批次的补齐长度变化，同一个句子放在不同批次里贡献不同。忽略 PAD 后，loss 是真实 token 上的平均交叉熵。
测试集用来模拟"模型从没见过的新数据"。如果用测试集建词表，模型就提前知道了测试集里有哪些词，未登录词率被人为压到 0，评估出的效果会比真实部署时更乐观，这属于数据泄漏。这一点在本实验中确实影响很大：测试集有 35% 的句子含未登录词，这类句子的 BLEU 明显更低。

**为什么只看训练 loss 或少数成功样例，不能说明翻译质量？**
训练 loss 衡量的是对**训练集**的拟合程度。本实验从第 11 轮到第 14 轮，训练 loss 从 1.58 降到 1.18，验证 loss 却从 2.90 升到 2.93，训练 loss 越低反而越过拟合。此外，训练 loss 是在 Teacher Forcing 下计算的，不反映自回归生成时的错误累积。少数成功样例则存在挑选偏差：短句、高频句很容易翻对（例如 `où est le beurre ?` → `where ' s the …`），但测试集 1000 句中只有 3 句与参考完全一致。必须在**未参与训练和模型选择**的完整测试集上，用自动指标（BLEU）评估，同时系统地检查随机样例和失败案例，并记住 BLEU 本身也会低估同义表达。

## 8. 局限与可能的改进

- **Attention**：让解码器每一步都能回看编码器所有时间步的输出，预计能显著缓解长句和漏译问题。
- **未知词**：设置 `min_freq=2`，让训练集低频词变成 `<UNK>`，从而训练出 `<UNK>` 的向量；或者改用子词（BPE），把 `sentais` 拆成可以共享的片段。
- **过拟合**：加 Dropout、权重衰减，或使用更多训练数据（合格句对共 149,414 对，本实验按要求只用了 10,000 对）。
- **解码**：用 Beam Search 代替贪心解码，可减少重复和局部错误。
- **曝光偏差**：使用逐步降低 Teacher Forcing 比例的 Scheduled Sampling。

## 9. 复现

```bash
cd nmt_lstm
.venv/bin/python check_lesson1.py   # 数据处理自检，12/12
.venv/bin/python train.py           # 训练，约 40 秒，输出 outputs/best_model.pt、loss_curve.png、history.json
.venv/bin/python evaluate.py        # 测试 BLEU、10 条样例、失败案例候选，输出 outputs/test_*.json*
```

环境：Python 3.14、PyTorch 2.14.1（CPU）、sacreBLEU 2.6.0、matplotlib 3.11.2。
