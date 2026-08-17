"""
Trains the DeepSC semantic communication model (deepsc_model.py) on the
synthetic corpus (deepsc_corpus.py) and evaluates it across a grid of SINR
values to produce deepsc_lookup_table.csv: the SINR(dB) -> semantic
similarity / word accuracy table consumed at runtime by semantic_node.py
via deepsc_lookup.py.

This script is the only place in Week9 work/ that needs torch + a training
loop; the system-level ISAC simulation only reads the resulting CSV.

Usage:
    python "Week9 work/train_deepsc.py"
"""

import csv
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .deepsc_corpus import build_corpus, train_val_split
    from .deepsc_model import DeepSC, Vocabulary
except ImportError:
    from deepsc_corpus import build_corpus, train_val_split
    from deepsc_model import DeepSC, Vocabulary

SEED = 0
MAX_LEN = 16
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
CHANNEL_DIM = 16
BATCH_SIZE = 32
EPOCHS = 120
LR = 1e-3
# Training across an unrealistically wide SNR range (e.g. down to -100 dB)
# was tried and rejected: at such SNRs no scheme can recover the signal, so
# those batches contribute near-random gradients that destabilize the shared
# encoder/decoder weights and hurt convergence even at high SNR (accuracy at
# 20 dB dropped from ~1.0 to ~0.37). -20 dB is already a near-total-failure
# regime for this model/corpus, so the table's flat extrapolation below it
# is a physically reasonable "channel unusable" floor, not an artifact.
TRAIN_SNR_RANGE_DB = (-20.0, 20.0)
EVAL_SNR_GRID_DB = list(range(-20, 21, 2))
EVAL_TRIALS_PER_SNR = 8

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOOKUP_TABLE_PATH = os.path.join(THIS_DIR, "deepsc_lookup_table.csv")
CHECKPOINT_PATH = os.path.join(THIS_DIR, "deepsc_model.pt")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_batches(sentences, vocab: Vocabulary, batch_size: int, device):
    ids = [vocab.encode(s, MAX_LEN) for s in sentences]
    ids = torch.tensor(ids, dtype=torch.long, device=device)
    for i in range(0, ids.size(0), batch_size):
        yield ids[i : i + batch_size]


def shift_for_teacher_forcing(src_ids: torch.Tensor, vocab: Vocabulary):
    """src_ids already end with <end> (+ padding). Builds teacher-forced
    decoder input (<start> + tokens) and target output (tokens + <end>)."""
    batch, seq_len = src_ids.shape
    start_col = torch.full((batch, 1), vocab.start_id, dtype=torch.long, device=src_ids.device)
    tgt_in = torch.cat([start_col, src_ids[:, :-1]], dim=1)
    tgt_out = src_ids
    return tgt_in, tgt_out


def train_model(model: DeepSC, vocab: Vocabulary, train_sentences, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for epoch in range(1, EPOCHS + 1):
        random.shuffle(train_sentences)
        total_loss, n_batches = 0.0, 0
        for batch in make_batches(train_sentences, vocab, BATCH_SIZE, device):
            tgt_in, tgt_out = shift_for_teacher_forcing(batch, vocab)
            snr_db = torch.empty(batch.size(0), device=device).uniform_(*TRAIN_SNR_RANGE_DB)

            logits = model(batch, tgt_in, snr_db)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1),
                ignore_index=vocab.pad_id,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{EPOCHS}  loss={total_loss / max(n_batches, 1):.4f}")


@torch.no_grad()
def word_accuracy(pred_ids: torch.Tensor, ref_ids: torch.Tensor, vocab: Vocabulary) -> float:
    correct, total = 0, 0
    for pred_row, ref_row in zip(pred_ids.tolist(), ref_ids.tolist()):
        ref_words = []
        for t in ref_row:
            if t in (vocab.end_id, vocab.pad_id):
                break
            ref_words.append(t)
        pred_words = []
        for t in pred_row:
            if t in (vocab.end_id, vocab.pad_id):
                break
            pred_words.append(t)
        n = max(len(ref_words), 1)
        matches = sum(1 for a, b in zip(pred_words, ref_words) if a == b)
        correct += matches
        total += n
    return correct / max(total, 1)


def truncate_after_end(ids: torch.Tensor, vocab: Vocabulary) -> torch.Tensor:
    """Zeroes out (pads) everything from the first <end> token onward, per
    row, so the embedding pooling mask lines up with word_accuracy's
    definition of 'the sentence' rather than including decoder artifacts
    the model kept emitting past <end>."""
    out = ids.clone()
    for row in range(out.size(0)):
        end_positions = (out[row] == vocab.end_id).nonzero(as_tuple=True)[0]
        if end_positions.numel() > 0:
            cut = end_positions[0].item()
            out[row, cut:] = vocab.pad_id
    return out


@torch.no_grad()
def evaluate_at_snr(model: DeepSC, vocab: Vocabulary, val_ids: torch.Tensor, snr_db: float):
    model.eval()
    clean_emb = F.normalize(model.sentence_embedding(val_ids), dim=-1)

    sims, accs = [], []
    for _ in range(EVAL_TRIALS_PER_SNR):
        pred_ids = model.greedy_decode(val_ids, snr_db, vocab, MAX_LEN)
        pred_padded = F.pad(pred_ids, (0, MAX_LEN - pred_ids.size(1)), value=vocab.pad_id)
        pred_padded = truncate_after_end(pred_padded, vocab)
        pred_emb = F.normalize(model.sentence_embedding(pred_padded), dim=-1)

        cos_sim = (clean_emb * pred_emb).sum(dim=-1).clamp(-1.0, 1.0)
        sims.append(cos_sim.mean().item())
        accs.append(word_accuracy(pred_ids, val_ids, vocab))

    return float(np.mean(sims)), float(np.mean(accs))


def build_lookup_table(model: DeepSC, vocab: Vocabulary, val_sentences, device):
    val_ids = torch.tensor([vocab.encode(s, MAX_LEN) for s in val_sentences], dtype=torch.long, device=device)

    rows = []
    print("\n  SINR(dB) | Semantic Similarity | Word Accuracy")
    for snr_db in EVAL_SNR_GRID_DB:
        sim, acc = evaluate_at_snr(model, vocab, val_ids, snr_db)
        # Cosine similarity of embeddings lies in [-1, 1]; map to a [0, 1]
        # semantic-fidelity score, consistent with the rest of the system
        # model where semantic_similarity in [0, 1].
        similarity_01 = float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))
        rows.append((snr_db, similarity_01, acc))
        print(f"  {snr_db:8.1f} | {similarity_01:19.4f} | {acc:13.4f}")

    with open(LOOKUP_TABLE_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sinr_db", "semantic_similarity", "word_accuracy"])
        writer.writerows(rows)
    print(f"\nSaved DeepSC lookup table -> {LOOKUP_TABLE_PATH}")


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sentences = build_corpus(num_sentences=400, seed=42)
    train_sentences, val_sentences = train_val_split(sentences, val_ratio=0.2, seed=7)
    vocab = Vocabulary(sentences)
    print(f"Corpus: {len(sentences)} sentences ({len(train_sentences)} train / {len(val_sentences)} val), "
          f"vocab size = {len(vocab)}")

    model = DeepSC(
        vocab_size=len(vocab), d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD, channel_dim=CHANNEL_DIM, max_len=MAX_LEN,
        pad_id=vocab.pad_id,
    ).to(device)

    print(f"\nTraining DeepSC for {EPOCHS} epochs on {device}...")
    train_model(model, vocab, train_sentences, device)

    print("\nEvaluating trained DeepSC model across the SINR grid...")
    build_lookup_table(model, vocab, val_sentences, device)

    torch.save({"model_state": model.state_dict(), "vocab": vocab.token_to_id,
                "config": dict(d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
                                dim_feedforward=DIM_FEEDFORWARD, channel_dim=CHANNEL_DIM,
                                max_len=MAX_LEN, pad_id=vocab.pad_id)},
               CHECKPOINT_PATH)
    print(f"Saved trained DeepSC checkpoint -> {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
