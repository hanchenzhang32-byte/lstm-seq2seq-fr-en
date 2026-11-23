"""Training.

- Loss: CrossEntropyLoss(ignore_index=pad_id) on raw logits
- Full teacher forcing (ratio 1.0)
- Train / validation loss logged every epoch, averaged over non-PAD tokens
  (validation loss also uses teacher forcing)
- Best checkpoint chosen on validation loss only, with early stopping; the test split is never touched here

Run: python train.py
"""

import json
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from data_prep import load_pairs, split_pairs
from dataset import Vocab, make_loader
from model import Seq2Seq

CONFIG = {
    "emb_dim": 128,
    "hidden_dim": 128,
    "num_layers": 1,
    "batch_size": 32,
    "lr": 1e-3,
    "max_epochs": 20,
    "patience": 3,            # stop after 3 epochs without a new best validation loss
    "clip_norm": 1.0,
    "teacher_forcing": 1.0,
    "seed": 42,
}
CKPT_PATH = "outputs/best_model.pt"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, optimizer=None, clip_norm=None):
    """Evaluation only when optimizer is None. Returns the mean loss per non-PAD token."""
    training = optimizer is not None
    model.train(training)
    total_loss, total_tokens = 0.0, 0
    with torch.set_grad_enabled(training):
        for src, src_lens, tgt_in, tgt_out in loader:
            logits = model(src, src_lens, tgt_in)                       # (B, T, V)
            # CrossEntropyLoss expects (N, V) and (N,): flatten batch and time
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            n_tokens = (tgt_out != criterion.ignore_index).sum().item()

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()

            # loss is the batch mean over non-PAD tokens; weight it back by token count
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    return total_loss / total_tokens


def plot_curves(history, best_epoch, path):
    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [h["train_loss"] for h in history], "o-", label="train loss")
    plt.plot(epochs, [h["val_loss"] for h in history], "s-", label="val loss (teacher forcing)")
    plt.axvline(best_epoch, color="gray", linestyle="--", label=f"best checkpoint: epoch {best_epoch}")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy per non-PAD token")
    plt.title("LSTM Seq2Seq fr→en")
    plt.xticks(epochs)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)


def main():
    set_seed(CONFIG["seed"])
    train, val, _test = split_pairs(load_pairs())

    # vocabularies come from the training split only
    fr_vocab = Vocab([fr for fr, _ in train])
    en_vocab = Vocab([en for _, en in train])
    train_loader = make_loader(train, fr_vocab, en_vocab, CONFIG["batch_size"], shuffle=True, seed=CONFIG["seed"])
    val_loader = make_loader(val, fr_vocab, en_vocab, CONFIG["batch_size"], shuffle=False)

    model = Seq2Seq(len(fr_vocab), len(en_vocab), CONFIG["emb_dim"], CONFIG["hidden_dim"],
                    fr_vocab.pad_id, en_vocab.pad_id)
    criterion = nn.CrossEntropyLoss(ignore_index=en_vocab.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"French vocab {len(fr_vocab)}, English vocab {len(en_vocab)}, parameters {n_params:,}")

    history, best_val, best_epoch, bad_epochs = [], float("inf"), 0, 0
    for epoch in range(1, CONFIG["max_epochs"] + 1):
        start = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, CONFIG["clip_norm"])
        val_loss = run_epoch(model, val_loader, criterion)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        mark = ""
        if val_loss < best_val:
            best_val, best_epoch, bad_epochs = val_loss, epoch, 0
            torch.save({
                "model_state": model.state_dict(),
                "fr_itos": fr_vocab.itos,
                "en_itos": en_vocab.itos,
                "config": CONFIG,
                "epoch": epoch,
                "val_loss": val_loss,
            }, CKPT_PATH)
            mark = "  <- best, saved"
        else:
            bad_epochs += 1
        print(f"epoch {epoch:2d}  train {train_loss:.4f}  val {val_loss:.4f}  ({time.time() - start:.1f}s){mark}")
        if bad_epochs >= CONFIG["patience"]:
            print(f"No improvement in validation loss for {CONFIG['patience']} epochs; stopping early")
            break

    with open("outputs/history.json", "w", encoding="utf-8") as f:
        json.dump({"config": CONFIG, "best_epoch": best_epoch, "best_val_loss": best_val,
                   "history": history}, f, indent=2)
    plot_curves(history, best_epoch, "outputs/loss_curve.png")
    print(f"Best epoch {best_epoch}, validation loss {best_val:.4f}; saved {CKPT_PATH} and outputs/loss_curve.png")


if __name__ == "__main__":
    main()
