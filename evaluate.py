"""Inference and evaluation.

- translate(sentence): raw French sentence → English (greedy decoding, own predictions fed back, ≤15 tokens)
- Corpus BLEU on the full test split (sacreBLEU, tokenize="none", no teacher forcing)
- 10 test examples drawn with a fixed seed
- Rule-based failure-case candidates (repetition, unknown words, omission, long sentences)

Run: python evaluate.py
"""

import json
import random

import sacrebleu
import torch
from sacrebleu.metrics import BLEU

from data_prep import load_pairs, split_pairs, tokenize
from dataset import PAD, SOS, EOS, Vocab
from model import Seq2Seq

CKPT_PATH = "outputs/best_model.pt"
MAX_GEN_LEN = 15


def vocab_from_itos(itos):
    vocab = Vocab([])
    vocab.itos = list(itos)
    vocab.stoi = {tok: i for i, tok in enumerate(itos)}
    return vocab


def load_model(path=CKPT_PATH):
    ckpt = torch.load(path, map_location="cpu")
    fr_vocab, en_vocab = vocab_from_itos(ckpt["fr_itos"]), vocab_from_itos(ckpt["en_itos"])
    cfg = ckpt["config"]
    model = Seq2Seq(len(fr_vocab), len(en_vocab), cfg["emb_dim"], cfg["hidden_dim"],
                    fr_vocab.pad_id, en_vocab.pad_id)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, fr_vocab, en_vocab, ckpt


def strip_specials(tokens):
    """Cut at the first <EOS>, drop <SOS>/<EOS>/<PAD>; keep <UNK>."""
    out = []
    for tok in tokens:
        if tok == EOS:
            break
        if tok not in (SOS, PAD):
            out.append(tok)
    return out


def translate_tokens(model, fr_tokens, fr_vocab, en_vocab):
    src = torch.tensor([fr_vocab.encode(fr_tokens)])
    src_lens = torch.tensor([len(fr_tokens)])
    model.eval()
    with torch.no_grad():
        ids = model.greedy_decode(src, src_lens, en_vocab.sos_id, en_vocab.eos_id, MAX_GEN_LEN)
    return strip_specials(en_vocab.decode(ids[0].tolist()))


_cache = {}


def translate(sentence: str) -> str:
    """Translate one raw French sentence, using exactly the training preprocessing."""
    if not _cache:
        _cache["model"], _cache["fr"], _cache["en"], _ = load_model()
    fr_tokens = tokenize(sentence)
    return " ".join(translate_tokens(_cache["model"], fr_tokens, _cache["fr"], _cache["en"]))


def has_repetition(tokens):
    if any(a == b for a, b in zip(tokens, tokens[1:])):
        return True
    bigrams = list(zip(tokens, tokens[1:]))
    return len(bigrams) != len(set(bigrams))


def main():
    model, fr_vocab, en_vocab, ckpt = load_model()
    _train, _val, test = split_pairs(load_pairs())

    rows = []
    for fr, en in test:
        pred = translate_tokens(model, fr, fr_vocab, en_vocab)
        rows.append({
            "fr": " ".join(fr),
            "ref": " ".join(en),                     # reference: original tokens, OOV words not replaced by <UNK>
            "pred": " ".join(pred),
            "fr_unk": [t for t in fr if t not in fr_vocab.stoi],
            "fr_len": len(fr),
            "ref_len": len(en),
            "pred_len": len(pred),
        })

    # force=True only silences sacreBLEU's "looks tokenized" warning; scoring tokenized text is intended
    bleu = BLEU(tokenize="none", force=True)
    score = bleu.corpus_score([r["pred"] for r in rows], [[r["ref"] for r in rows]])
    signature = str(bleu.get_signature())
    sent_bleu = BLEU(tokenize="none", effective_order=True)
    for r in rows:
        r["sent_bleu"] = sent_bleu.sentence_score(r["pred"], [r["ref"]]).score

    print(f"Best checkpoint: epoch {ckpt['epoch']}, validation loss {ckpt['val_loss']:.4f}")
    print(f"Test corpus BLEU = {score.score:.2f}")
    print(f"  {score}")
    print(f"  sacreBLEU {sacrebleu.__version__}, signature: {signature}")
    exact = sum(r["pred"] == r["ref"] for r in rows)
    n_unk_out = sum("<UNK>" in r["pred"] for r in rows)
    print(f"  exact matches {exact}/{len(rows)}; outputs containing <UNK>: {n_unk_out}")

    print("\n===== 10 test examples (random.Random(42).sample) =====")
    sample_idx = sorted(random.Random(42).sample(range(len(rows)), 10))
    for k, i in enumerate(sample_idx, 1):
        r = rows[i]
        print(f"[{k}] #{i}\n  FR : {r['fr']}\n  REF: {r['ref']}\n  OUT: {r['pred']}")

    print("\n===== Failure-case candidates =====")
    categories = {
        "Repetition": [r for r in rows if has_repetition(r["pred"].split())],
        "Unknown source words": [r for r in rows if r["fr_unk"] and r["sent_bleu"] < 20],
        "Omission (output ≥3 tokens shorter than reference)": [r for r in rows if r["ref_len"] - r["pred_len"] >= 3],
        "Long sentence (10 French tokens) with BLEU < 10": [r for r in rows if r["fr_len"] == 10 and r["sent_bleu"] < 10],
    }
    for name, cands in categories.items():
        print(f"\n--- {name}: {len(cands)} total, first 6 ---")
        for r in cands[:6]:
            unk = f"  OOV={r['fr_unk']}" if r["fr_unk"] else ""
            print(f"  FR : {r['fr']}{unk}\n  REF: {r['ref']}\n  OUT: {r['pred']}\n")

    print("===== translate() demo =====")
    for s in ["Je suis fatigué.", "Il fait très froid aujourd’hui !", "Où est la gare ?"]:
        print(f"  {s!r} → {translate(s)!r}")

    with open("outputs/test_translations.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open("outputs/test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"bleu": score.score, "bleu_full": str(score), "signature": signature,
                   "sacrebleu_version": sacrebleu.__version__, "best_epoch": ckpt["epoch"],
                   "sample_indices": sample_idx}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
