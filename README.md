# LSTM Seq2Seq: French → English Translation

A from-scratch PyTorch encoder–decoder (LSTM, no attention) that translates short French sentences into English, trained on 8,000 Tatoeba sentence pairs. The focus is on a clean, reproducible pipeline and an honest evaluation: validation-only model selection, a held-out test set, sacreBLEU with a published signature, and an analysis of where and why the model fails.

## Results

| | |
|---|---|
| Test corpus BLEU | **11.39** (1,000 sentences, greedy decoding, no teacher forcing) |
| Best checkpoint | epoch 11 of 14 (validation loss 2.9008; early stopping) |
| Model | 2.05 M parameters, 128-d embeddings and hidden state |
| Training time | ~40 s on CPU; identical losses on rerun |
| sacreBLEU signature | `nrefs:1\|case:mixed\|eff:no\|tok:none\|smooth:exp\|version:2.6.0` |

![Loss curves](outputs/loss_curve.png)

| French | Reference | Model output |
|---|---|---|
| personne ne te demande de manger ça . | no one is asking you to eat that . | no one would you like to do that . |
| vous ne pouvez pas avoir trente ans ! | you can ' t be thirty years old . | you can ' t have a little . |
| je suis à l ' extérieur . | i ' m outside . | i ' m in the room . |

The model reliably gets the sentence frame right (subject, negation, question form) but often substitutes content words with frequent training phrases. Key findings from the [full report](report.md):

- **Unknown words hurt most.** 35% of test sentences contain a French word never seen in training; they score 8.73 BLEU against 12.68 for the rest. Because every training word is kept, the `<UNK>` embedding is never trained.
- **Longer sentences degrade.** BLEU falls from 17.45 (1–5 tokens) to 10.28 (9–10 tokens): the whole sentence has to fit into one 128-d state.
- **Exposure bias shows up as repetition** (`it ' s really really good`), since training always feeds the decoder the correct previous word.
- **BLEU underrates some valid outputs**, e.g. `sa maison` → `his house` where the reference says `her house`; French `sa` does not mark the owner's gender.

## How it works

```
French ids → Embedding → LSTM ──(h, c)──▶ LSTM → Linear → English logits
                encoder                      decoder (previous English token in)
```

- **`data_prep.py`** — Unicode normalisation, regex tokenisation, length filter (≤10 tokens), French-side deduplication, seeded 8k / 1k / 1k split
- **`dataset.py`** — training-only vocabularies, `Dataset` / `DataLoader` with in-batch padding, true lengths and one-step-shifted targets
- **`model.py`** — encoder with `pack_padded_sequence` so padding never reaches the final state; decoder initialised from the encoder's `(h, c)`; greedy decoding
- **`train.py`** — token-averaged cross-entropy ignoring `<PAD>`, Adam, gradient clipping, validation-based checkpointing and early stopping
- **`evaluate.py`** — `translate(sentence)`, corpus BLEU, seeded sample, rule-based failure-case mining
- **`tests/`** — pytest checks for normalisation, tokenisation, filtering, deduplication and the split

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

Download the data (ManyThings.org rejects requests without a browser User-Agent; you can also download it in a browser and unzip into `data/`):

```bash
mkdir -p data
curl -L -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36" -o data/fra-eng.zip https://www.manythings.org/anki/fra-eng.zip
unzip -o data/fra-eng.zip -d data
```

```bash
python -m pytest     # data-processing tests
python train.py      # → outputs/best_model.pt, loss_curve.png, history.json
python evaluate.py   # → test BLEU, examples, failure cases, outputs/test_*.json*
```

```python
>>> from evaluate import translate
>>> translate("Où est la gare ?")
"where ' s the door ?"
```

The results above use the 2026-02-13 release of `fra.txt` (240,521 lines). ManyThings.org updates the file, so a newer download will give slightly different counts and scores.

## Next steps

Attention, subword tokenisation (BPE) for unknown words, `min_freq=2` so `<UNK>` gets trained, dropout against the overfitting visible after epoch 9, and beam search. See [report.md §8](report.md#8-limitations-and-next-steps).

## Data

Tatoeba French–English pairs packaged by [ManyThings.org](https://www.manythings.org/anki/), licensed CC-BY 2.0 (France). Attribution: www.manythings.org/anki and tatoeba.org. The raw data is not included in this repository.
