"""第 3 课：不带 Attention 的 Encoder–Decoder

Encoder：法语编号 → Embedding → LSTM → 最终 (h, c)
Decoder：上一个英语词 → Embedding → LSTM（初始状态 = 编码器的 (h, c)）→ Linear → 英语词表 logits
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=1, batch_first=True)

    def forward(self, src, src_lens):
        emb = self.embedding(src)                                   # (B, S, E)
        # pack：按真实长度告诉 LSTM 每句在哪里结束，
        # 这样最终的 (h, c) 取自每句最后一个真实词，而不是读完一串 <PAD> 之后的状态
        packed = pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, c) = self.lstm(packed)                               # h, c: (1, B, H)
        return h, c


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=1, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_in, h, c):
        """tgt_in: (B, T) 每一步的“上一个词”。可以一次喂整句（Teacher Forcing），也可以每次喂 1 个词（推理）。"""
        emb = self.embedding(tgt_in)                                # (B, T, E)
        output, (h, c) = self.lstm(emb, (h, c))                     # output: (B, T, H)
        logits = self.out(output)                                   # (B, T, V_en)，原始 logits，不做 softmax
        return logits, h, c


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_dim, hidden_dim, src_pad_id, tgt_pad_id):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, emb_dim, hidden_dim, src_pad_id)
        self.decoder = Decoder(tgt_vocab_size, emb_dim, hidden_dim, tgt_pad_id)

    def forward(self, src, src_lens, tgt_in):
        """训练用：完整 Teacher Forcing。

        解码器每一步的输入都是真实的上一个英语词（tgt_in 第 t 列），
        与“一步一步循环、每步喂真实词”完全等价，只是让 LSTM 一次处理整句更快。
        """
        h, c = self.encoder(src, src_lens)
        logits, _, _ = self.decoder(tgt_in, h, c)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src, src_lens, sos_id, eos_id, max_len=15):
        """推理用：从 <SOS> 开始，每步取概率最大的词，作为下一步输入。不使用任何参考译文。"""
        h, c = self.encoder(src, src_lens)
        batch_size = src.size(0)
        prev = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=src.device)
        outputs = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits, h, c = self.decoder(prev, h, c)                 # (B, 1, V)
            prev = logits.argmax(dim=-1)                            # (B, 1) 自己的预测成为下一步输入
            outputs.append(prev)
            finished |= prev.squeeze(1) == eos_id
            if finished.all():
                break
        return torch.cat(outputs, dim=1)                            # (B, ≤max_len)
