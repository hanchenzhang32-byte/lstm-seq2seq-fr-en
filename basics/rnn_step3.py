import torch
import torch.nn as nn

torch.manual_seed(0)

BATCH = 32
SEQ_LEN = 10
INPUT_SIZE = 128
HIDDEN_SIZE = 128

rnn = nn.RNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True)
x = torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)

output, h_last = rnn(x)
print("output.shape = ", output.shape)
print("h_last.shape = ", h_last.shape)
print("output = ", output.squeeze())