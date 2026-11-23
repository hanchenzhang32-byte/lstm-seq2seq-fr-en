import math

import torch
import torch.nn as nn

torch.manual_seed(1)
rnn = nn.RNN(input_size=1,hidden_size=1,batch_first=True)

print("w_x = ", rnn.weight_ih_l0.item())
print("w_h = ", rnn.weight_hh_l0.item())


# ③ 同一句话 [1, 0, 0, 0, 0]
sentence = torch.tensor([[[1.0], [0.0], [0.0], [0.0], [0.0]]])
output, h_last = rnn(sentence)
print(output.squeeze())