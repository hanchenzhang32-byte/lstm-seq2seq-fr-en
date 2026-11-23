"""前置第 1 节：文字怎么变成神经网络能吃的数字

MLP / CNN 的输入本来就是数字（像素、特征）。文字不是，所以要三步：
    "i am happy"  →  ["i", "am", "happy"]  →  [4, 5, 6]  →  三个向量
        句子             token 列表           编号            词向量

运行：.venv/bin/python basics/01_text_to_numbers.py
"""

import torch
import torch.nn as nn

torch.manual_seed(0)


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


# ---------------------------------------------------------------
section("1. 建词表：每个不同的 token 分配一个整数编号")
# ---------------------------------------------------------------
sentences = [
    ["i", "am", "happy", "."],
    ["i", "am", "tired", "."],
    ["you", "are", "happy", "."],
]

# 前 4 个编号留给特殊 token（作业要求的那 4 个），真实单词从 4 开始编号
vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
for sent in sentences:
    for tok in sent:
        if tok not in vocab:
            vocab[tok] = len(vocab)        # 新词的编号 = 当前词表大小
print("词表:", vocab)

id_to_token = {i: t for t, i in vocab.items()}   # 反向表：编号 → 词，翻译输出时要用


def encode(tokens):
    # 不在词表里的词（未登录词）统一映射为 <UNK>
    return [vocab.get(t, vocab["<UNK>"]) for t in tokens]


print("encode(['i', 'am', 'happy', '.'])   =", encode(["i", "am", "happy", "."]))
print("encode(['i', 'am', 'hungry', '.'])  =", encode(["i", "am", "hungry", "."]), " ← hungry 没见过，变成 3 (<UNK>)")


# ---------------------------------------------------------------
section("2. 方法一：one-hot（你在 MLP 里见过的分类标签编码）")
# ---------------------------------------------------------------
V = len(vocab)
ids = torch.tensor(encode(["i", "am", "happy"]))
one_hot = nn.functional.one_hot(ids, num_classes=V).float()
print(f"词表大小 V = {V}，每个词变成一个长度 {V} 的向量，只有一个位置是 1：")
print(one_hot)
print("问题 1：真实词表有几千上万个词，向量又长又稀疏。")
cos = nn.functional.cosine_similarity
print(f"问题 2：任意两个不同词的相似度都是 0 —— sim(happy, tired) = {cos(one_hot[2], nn.functional.one_hot(torch.tensor(vocab['tired']), V).float(), dim=0):.1f}，"
      "模型无法知道 happy 和 tired 都是形容词。")


# ---------------------------------------------------------------
section("3. 方法二：nn.Embedding —— 一张可训练的查找表")
# ---------------------------------------------------------------
EMB_DIM = 3                                   # 作业里是 128，这里用 3 方便看
emb = nn.Embedding(num_embeddings=V, embedding_dim=EMB_DIM)
print(f"emb.weight 形状 = {tuple(emb.weight.shape)}  ← 就是一个 V × {EMB_DIM} 的矩阵，第 i 行是编号 i 的词向量")
print("emb(ids) 就是按编号取行：")
print("  ids        =", ids.tolist())
print("  emb(ids)   =\n", emb(ids).detach())
print("  emb.weight[ids] 完全相同？", torch.equal(emb(ids), emb.weight[ids]))

# 和 MLP 的联系：one-hot 乘以权重矩阵，结果正好“选出”对应的那一行
# 所以 Embedding 等价于“one-hot 输入 + 没有 bias 的 Linear 层”，只是直接查表更省计算
print("  one_hot @ weight 也相同？", torch.allclose(one_hot @ emb.weight, emb(ids)))
print("关键：这些向量一开始是随机的，训练时和其他参数一样被梯度更新，")
print("      最后意思相近的词会自动被推到相近的位置。")


# ---------------------------------------------------------------
section("4. 一个批次：句子长度不同，要用 <PAD> 补齐")
# ---------------------------------------------------------------
batch_tokens = [
    ["i", "am", "happy", "."],
    ["you", "are", "."],
    ["happy", "."],
]
batch_ids = [encode(s) for s in batch_tokens]
lengths = [len(s) for s in batch_ids]        # 真实长度，后面 LSTM 会用到
max_len = max(lengths)
padded = [s + [vocab["<PAD>"]] * (max_len - len(s)) for s in batch_ids]
x = torch.tensor(padded)
print("补齐前:", batch_ids)
print("补齐后张量 x =\n", x)
print("真实长度 lengths =", lengths)
print(f"x 的形状 = {tuple(x.shape)}  即 (batch_size, seq_len)")

emb_pad = nn.Embedding(V, EMB_DIM, padding_idx=vocab["<PAD>"])
out = emb_pad(x)
print(f"\nemb(x) 的形状 = {tuple(out.shape)}  即 (batch_size, seq_len, emb_dim)")
print("  —— 每个编号都被换成了一个向量，所以多出最后一维")
print("padding_idx=0 的效果：<PAD> 的向量固定为全 0，且不会被训练：")
print("  第 3 句（happy . <PAD> <PAD>）的向量 =\n", out[2].detach())
