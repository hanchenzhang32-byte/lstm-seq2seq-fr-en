"""前置第 1 节补充：把“编号 → 128 维向量”画出来

运行：.venv/bin/python basics/01b_embedding_visual.py
输出：basics/embedding_visual.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(0)

vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "i", "am", "happy", ".", "tired", "you", "are"]
emb = nn.Embedding(len(vocab), 128, padding_idx=0)
W = emb.weight.detach().numpy()

sentence = [("i", 4), ("am", 5), ("hungry", 3), (".", 7)]

fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1, 1.2], hspace=0.55, wspace=0.3)

# ① 整张表：11 行 × 128 列
ax = fig.add_subplot(gs[0, :])
ax.imshow(W, cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
ax.set_yticks(range(len(vocab)), [f"{i}  {w}" for i, w in enumerate(vocab)])
ax.set_xlabel("128 个维度（列）")
ax.set_title("① Embedding 表 emb.weight：形状 (11, 128)。每一行 = 一个词的向量（蓝=负，红=正，<PAD> 行全 0）")
for _, idx in sentence:
    ax.add_patch(plt.Rectangle((-0.5, idx - 0.5), 128, 1, fill=False, ec="black", lw=2))

# ② 句子里每个词：编号 → 取出那一行，排成 8×16
for k, (word, idx) in enumerate(sentence):
    ax = fig.add_subplot(gs[1, k])
    ax.imshow(W[idx].reshape(8, 16), cmap="coolwarm", vmin=-2, vmax=2)
    ax.set_xticks([]); ax.set_yticks([])
    note = "（没见过 → 取 <UNK> 行）" if word == "hungry" else ""
    ax.set_title(f'② "{word}" → 编号 {idx} → 第 {idx} 行\n128 个数排成 8×16 {note}', fontsize=10)

# ③ 意义：训练前随机 vs 训练后（示意）按词性聚集
groups = {"代词": (["i", "you", "he", "we"], "#7F77DD", (-3, 2)),
          "动词": (["am", "are", "is", "eat"], "#1D9E75", (3, 2.5)),
          "形容词": (["happy", "tired", "sad", "hungry"], "#D85A30", (2.5, -2.5)),
          "标点": ([".", "?", "!"], "#888780", (-3, -2.5))}
rng = np.random.default_rng(1)
for col, title, trained in [(slice(0, 2), "③ 训练前：随机初始化，词乱撒，没有意义", False),
                            (slice(2, 4), "④ 训练后（示意）：用法相似的词被梯度推到一起", True)]:
    ax = fig.add_subplot(gs[2, col])
    for name, (words, color, center) in groups.items():
        for w in words:
            xy = np.array(center) + rng.normal(0, 1.0, 2) if trained else rng.uniform(-4.5, 4.5, 2)
            ax.scatter(*xy, color=color, s=80)
            ax.annotate(w, xy, xytext=(6, 4), textcoords="offset points", fontsize=11)
        ax.scatter([], [], color=color, label=name)
    ax.set_xlim(-5.5, 5.5); ax.set_ylim(-5, 5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title + "\n（128 维压到 2 维画出）", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)

fig.suptitle('"i am hungry ."  →  [4, 5, 3, 7]  →  4 个 128 维向量  →  形状 (1, 4, 128)', fontsize=15)
out = "basics/embedding_visual.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print("已保存", out)
