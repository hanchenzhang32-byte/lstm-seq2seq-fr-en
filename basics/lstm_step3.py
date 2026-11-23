import torch
import torch.nn as nn

torch.manual_seed(0)
lstm = nn.LSTM(input_size=1, hidden_size=1, batch_first=True)

# ① 看看盒子里有哪些权重
for name, p in lstm.named_parameters():
    print(name, tuple(p.shape))
print()

x = torch.tensor([
    [[1.0], [0.0], [0.0], [0.0], [0.0]],   # 第 1 句
    [[0.0], [0.0], [0.0], [0.0], [1.0]],   # 第 2 句
])

# ② 和 RNN 唯一的区别：多返回一个 c
with torch.no_grad():
    output, (h_last, c_last) = lstm(x)

print("output 形状:", tuple(output.shape))
print("h_last 形状:", tuple(h_last.shape))
print("c_last 形状:", tuple(c_last.shape))
print()
print("第 1 句每一步的 h:", output[0].squeeze().round(decimals=4))
print("第 1 句最后的 h:  ", h_last[0, 0].round(decimals=4))
print("第 1 句最后的 c:  ", c_last[0, 0].round(decimals=4))