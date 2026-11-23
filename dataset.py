"""第 2 课：词表、Dataset、DataLoader

    token 列表  →  编号列表  →  一个批次的张量（批内补齐 + 真实长度 + 目标错位）

目标序列错位一位：
    英语 tokens      i am happy .
    解码器输入   <SOS> i  am  happy .
    预测目标         i  am happy  .  <EOS>
"""

from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset

PAD, SOS, EOS, UNK = "<PAD>", "<SOS>", "<EOS>", "<UNK>"
SPECIALS = [PAD, SOS, EOS, UNK]      # 固定编号 0, 1, 2, 3


class Vocab:
    """token ↔ 编号 的双向映射。只能用训练集建立。"""

    def __init__(self, sentences, min_freq=1):
        counter = Counter(tok for sent in sentences for tok in sent)
        self.itos = list(SPECIALS)                      # index → string
        # 按频率从高到低、同频按字母序，保证每次建出的词表一模一样
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
        # 未登录词 → <UNK>
        return [self.stoi.get(tok, self.unk_id) for tok in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]


class TranslationDataset(Dataset):
    """每个样本：(法语编号列表, 英语编号列表)，此时还不补齐、不加特殊 token。"""

    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.samples = [(src_vocab.encode(fr), tgt_vocab.encode(en)) for fr, en in pairs]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_collate_fn(pad_id, sos_id, eos_id):
    """DataLoader 每凑够 batch_size 个样本，就调用一次 collate_fn 把它们拼成张量。"""

    def collate(batch):
        srcs, tgts = zip(*batch)
        src_lens = torch.tensor([len(s) for s in srcs])          # 法语真实长度（不含 PAD）
        max_src = int(src_lens.max())
        max_tgt = max(len(t) for t in tgts) + 1                  # +1 给 <SOS>/<EOS>

        src = torch.full((len(batch), max_src), pad_id)          # 先全部填 PAD
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
    generator = torch.Generator().manual_seed(seed)              # 固定每个 epoch 的打乱顺序
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
    print(f"法语词表 {len(fr_vocab)}，英语词表 {len(en_vocab)}（均含 4 个特殊 token）")
    for name, split in [("val", val), ("test", test)]:
        fr_unk = sum(t not in fr_vocab.stoi for fr, _ in split for t in fr)
        fr_all = sum(len(fr) for fr, _ in split)
        print(f"{name} 法语未登录词比例 {fr_unk / fr_all:.1%}")

    src, src_lens, tgt_in, tgt_out = next(iter(make_loader(train, fr_vocab, en_vocab, 4, shuffle=False)))
    print("src 形状", tuple(src.shape), "真实长度", src_lens.tolist())
    print("第 1 句法语:", fr_vocab.decode(src[0].tolist()))
    print("解码器输入:", en_vocab.decode(tgt_in[0].tolist()))
    print("预测目标:  ", en_vocab.decode(tgt_out[0].tolist()))
