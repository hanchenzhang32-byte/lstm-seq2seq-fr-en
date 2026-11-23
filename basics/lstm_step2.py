import math


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# 门的权重（真实模型里是训练出来的，这里先手动设定）
w_i, b_i = 10.0, -5.0    # 输入门：读到 1 就开，读到 0 就关
w_f, b_f = 0.0, 5.0      # 遗忘门：先设成一直接近 1（几乎不擦）
w_o, b_o = 0.0, 5.0      # 输出门：一直接近 1

c = 0.0
sentence = [0, 1, 0, 0, 0]   # 重要的词在第 2 个位置

for t, x in enumerate(sentence):
    i = sigmoid(w_i * x + b_i)   # 每读一个词，三个门都重新算一次
    f = sigmoid(w_f * x + b_f)
    o = sigmoid(w_o * x + b_o)
    new = math.tanh(x)
    c = f * c + i * new          # 这两行和 step1 完全一样
    h = o * math.tanh(c)
    print(f"t={t}  x={x}  i={i:.3f}  f={f:.3f}  c={c:.4f}  h={h:.4f}")