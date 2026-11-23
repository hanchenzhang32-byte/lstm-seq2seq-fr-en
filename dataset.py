"""Vocabulary, Dataset and DataLoader.

    token lists → id lists → padded batch tensors (+ true source lengths, shifted targets)

Targets are shifted by one position:
    English tokens        i am happy .
    decoder input   <SOS> i  am  happy .
    target                i  am happy  .  <EOS>
"""

from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset

PAD, SOS, EOS, UNK = "<PAD>", "<SOS>", "<EOS>", "<UNK>"
SPECIALS = [PAD, SOS, EOS, UNK]      # fixed ids 0, 1, 2, 3


class Vocab:
    """Two-way token ↔ id mapping. Built from the training split only."""

    def __init__(self, sentences, min_freq=1):
        counter = Counter(tok for sent in sentences for tok in sent)
        self.itos = list(SPECIALS)                      # index → string
        # Frequency descending, ties alphabetical, so the vocabulary is deterministic
        for tok, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq >= min_freq:
                self.itos.append(tok)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}   # string → index

    def __len__(self):
        return len(self.itos)

    @property
    def pad_id(self):
        return self.stoi[PAD]

    @property
    def sos_id(self):
        return self.stoi[SOS]

    @property
    def eos_id(self):
        return self.stoi[EOS]

    @property
    def unk_id(self):
        return self.stoi[UNK]

    def encode(self, tokens):
        # out-of-vocabulary tokens → <UNK>
        return [self.stoi.get(tok, self.unk_id) for tok in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]


class TranslationDataset(Dataset):
    """Each item is (French ids, English ids), unpadded and without special tokens."""

    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.samples = [(src_vocab.encode(fr), tgt_vocab.encode(en)) for fr, en in pairs]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_collate_fn(pad_id, sos_id, eos_id):
    """Builds the function DataLoader uses to turn a list of samples into batch tensors."""

    def collate(batch):
        srcs, tgts = zip(*batch)
        src_lens = torch.tensor([len(s) for s in srcs])          # true French lengths
        max_src = int(src_lens.max())
        max_tgt = max(len(t) for t in tgts) + 1                  # +1 for <SOS>/<EOS>

        src = torch.full((len(batch), max_src), pad_id)
        tgt_in = torch.full((len(batch), max_tgt), pad_id)
        tgt_out = torch.full((len(batch), max_tgt), pad_id)
        for i, (s, t) in enumerate(batch):
            src[i, :len(s)] = torch.tensor(s)
            tgt_in[i, :len(t) + 1] = torch.tensor([sos_id] + t)   # <SOS> i am happy .
            tgt_out[i, :len(t) + 1] = torch.tensor(t + [eos_id])  # i am happy . <EOS>
        return src, src_lens, tgt_in, tgt_out

    return collate


def make_loader(pairs, src_vocab, tgt_vocab, batch_size, shuffle, seed=42):
    dataset = TranslationDataset(pairs, src_vocab, tgt_vocab)
    generator = torch.Generator().manual_seed(seed)              # reproducible epoch order
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=make_collate_fn(tgt_vocab.pad_id, tgt_vocab.sos_id, tgt_vocab.eos_id),
    )


if __name__ == "__main__":
    from data_prep import load_pairs, split_pairs

    train, val, test = split_pairs(load_pairs())
    fr_vocab = Vocab([fr for fr, _ in train])
    en_vocab = Vocab([en for _, en in train])
    print(f"French vocab {len(fr_vocab)}, English vocab {len(en_vocab)} (incl. 4 special tokens)")
    for name, split in [("val", val), ("test", test)]:
        fr_unk = sum(t not in fr_vocab.stoi for fr, _ in split for t in fr)
        fr_all = sum(len(fr) for fr, _ in split)
        print(f"{name} French OOV rate {fr_unk / fr_all:.1%}")

    src, src_lens, tgt_in, tgt_out = next(iter(make_loader(train, fr_vocab, en_vocab, 4, shuffle=False)))
    print("src shape", tuple(src.shape), "true lengths", src_lens.tolist())
    print("French:       ", fr_vocab.decode(src[0].tolist()))
    print("decoder input:", en_vocab.decode(tgt_in[0].tolist()))
    print("target:       ", en_vocab.decode(tgt_out[0].tolist()))
