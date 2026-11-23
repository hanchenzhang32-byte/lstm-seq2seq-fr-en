"""Encoder–decoder without attention.

Encoder: French ids → Embedding → LSTM → final (h, c)
Decoder: previous English token → Embedding → LSTM (initial state = encoder's (h, c)) → Linear → logits
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
        # Packing tells the LSTM where each sentence really ends, so the final (h, c)
        # comes from the last real token rather than from reading a run of <PAD>.
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
        """tgt_in: (B, T) previous tokens — a whole sentence (teacher forcing) or one step (inference)."""
        emb = self.embedding(tgt_in)                                # (B, T, E)
        output, (h, c) = self.lstm(emb, (h, c))                     # output: (B, T, H)
        logits = self.out(output)                                   # (B, T, V_en), raw logits
        return logits, h, c


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_dim, hidden_dim, src_pad_id, tgt_pad_id):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, emb_dim, hidden_dim, src_pad_id)
        self.decoder = Decoder(tgt_vocab_size, emb_dim, hidden_dim, tgt_pad_id)

    def forward(self, src, src_lens, tgt_in):
        """Training with full teacher forcing.

        Every decoder step receives the true previous English token (column t of tgt_in).
        Feeding the whole sequence at once is equivalent to a step-by-step loop, just faster.
        """
        h, c = self.encoder(src, src_lens)
        logits, _, _ = self.decoder(tgt_in, h, c)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src, src_lens, sos_id, eos_id, max_len=15):
        """Inference: start from <SOS>, take the argmax each step and feed it back. No reference is used."""
        h, c = self.encoder(src, src_lens)
        batch_size = src.size(0)
        prev = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=src.device)
        outputs = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits, h, c = self.decoder(prev, h, c)                 # (B, 1, V)
            prev = logits.argmax(dim=-1)                            # (B, 1) own prediction becomes next input
            outputs.append(prev)
            finished |= prev.squeeze(1) == eos_id
            if finished.all():
                break
        return torch.cat(outputs, dim=1)                            # (B, ≤max_len)
