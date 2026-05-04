"""Phase 3: Train neural network variants and compare accuracy."""
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import os

from neural_net import KanjiOnyomiModel, create_model

# ── Load data ───────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("dataset/kanji_dataset.csv")
with open("dataset/component_vocab.json") as f:
    component_to_id = json.load(f)
with open("dataset/onyomi_vocab.json") as f:
    onyomi_to_id = json.load(f)

id_to_onyomi = {v: k for k, v in onyomi_to_id.items()}

# Parse component IDs
df["comp_id_list"] = df["component_ids"].apply(
    lambda x: json.loads(x) if isinstance(x, str) else [])

# Filter to rows with on'yomi
df = df[df["onyomi"].notna() & (df["onyomi"] != "")].copy()
df["onyomi_id"] = df["onyomi"].map(onyomi_to_id)

# ── Build feature vocabularies ──────────────────────────────────────
# Pinyin initials
all_initials = sorted(set(df["pinyin_initial"].dropna().unique()) - {""})
pinyin_init_to_id = {p: i for i, p in enumerate(all_initials)}

# Pinyin finals
all_finals = sorted(set(df["pinyin_final"].dropna().unique()) - {""})
pinyin_final_to_id = {p: i for i, p in enumerate(all_finals)}

# Radicals
all_radicals = sorted(set(df["radical"].dropna().unique()) - {""})
radical_to_id = {r: i for i, r in enumerate(all_radicals)}

# Nasal coda mapping
nasal_to_vec = {"none": [1, 0, 0], "n": [0, 1, 0], "ng": [0, 0, 1]}

print(f"  Component vocab: {len(component_to_id)}")
print(f"  On'yomi vocab: {len(onyomi_to_id)}")
print(f"  Pinyin initials: {len(pinyin_init_to_id)}")
print(f"  Pinyin finals: {len(pinyin_final_to_id)}")
print(f"  Radicals: {len(radical_to_id)}")

# Save feature vocabularies for later use (lookup tool, etc.)
os.makedirs("dataset", exist_ok=True)
with open("dataset/pinyin_init_vocab.json", "w") as f:
    json.dump(pinyin_init_to_id, f, ensure_ascii=False)
with open("dataset/pinyin_final_vocab.json", "w") as f:
    json.dump(pinyin_final_to_id, f, ensure_ascii=False)
with open("dataset/radical_vocab.json", "w") as f:
    json.dump(radical_to_id, f, ensure_ascii=False)

# ── Prepare tensors ─────────────────────────────────────────────────
MAX_COMPS = 8  # max components per kanji

def prepare_features(data_df):
    """Convert dataframe rows to tensor features."""
    n = len(data_df)
    comp_ids = np.zeros((n, MAX_COMPS), dtype=np.int64)
    comp_mask = np.zeros((n, MAX_COMPS), dtype=np.bool_)
    pinyin_init_idx = np.zeros(n, dtype=np.int64)
    pinyin_final_idx = np.zeros(n, dtype=np.int64)
    pinyin_tone = np.zeros(n, dtype=np.int64)
    nasal_coda = np.zeros((n, 3), dtype=np.float32)
    radical_idx = np.zeros(n, dtype=np.int64)
    stroke_count = np.zeros(n, dtype=np.float32)
    onyomi_ids = np.zeros(n, dtype=np.int64)

    for i, (_, row) in enumerate(data_df.iterrows()):
        # Components
        clist = row["comp_id_list"]
        for j, cid in enumerate(clist[:MAX_COMPS]):
            comp_ids[i, j] = cid + 1  # shift by 1 for padding
            comp_mask[i, j] = True

        # Pinyin
        init = row["pinyin_initial"]
        if init and init in pinyin_init_to_id:
            pinyin_init_idx[i] = pinyin_init_to_id[init]
        final = row["pinyin_final"]
        if final and final in pinyin_final_to_id:
            pinyin_final_idx[i] = pinyin_final_to_id[final]
        tone = int(row["pinyin_tone"]) if not pd.isna(row["pinyin_tone"]) else 0
        pinyin_tone[i] = min(tone, 5)
        nasal = row["nasal_coda"]
        nasal_coda[i] = nasal_to_vec.get(nasal, [1, 0, 0])

        # Radical
        rad = row["radical"]
        if rad and rad in radical_to_id:
            radical_idx[i] = radical_to_id[rad]

        # Stroke count
        sc = row["stroke_count"]
        stroke_count[i] = sc / 30.0 if not pd.isna(sc) else 0  # normalize

        # Target
        onyomi_ids[i] = int(row["onyomi_id"]) if not pd.isna(row["onyomi_id"]) else 0

    return {
        "comp_ids": torch.tensor(comp_ids),
        "comp_mask": torch.tensor(comp_mask),
        "pinyin_init_idx": torch.tensor(pinyin_init_idx),
        "pinyin_final_idx": torch.tensor(pinyin_final_idx),
        "pinyin_tone": torch.tensor(pinyin_tone),
        "nasal_coda": torch.tensor(nasal_coda),
        "radical_idx": torch.tensor(radical_idx),
        "stroke_count": torch.tensor(stroke_count),
        "onyomi_ids": torch.tensor(onyomi_ids),
    }


class KanjiDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors["onyomi_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tensors.items()}


# ── Train/test split (stratified by JLPT level) ─────────────────────
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42,
    stratify=df["jlpt_level"] if len(df["jlpt_level"].unique()) > 1 else None
)
train_tensors = prepare_features(train_df)
test_tensors = prepare_features(test_df)

train_dataset = KanjiDataset(train_tensors)
test_dataset = KanjiDataset(test_tensors)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")


# ── Training ────────────────────────────────────────────────────────
def train_model(model, train_loader, test_loader, epochs=50, lr=0.001, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                batch["comp_ids"].to(device),
                batch["comp_mask"].to(device),
                batch["pinyin_init_idx"].to(device),
                batch["pinyin_final_idx"].to(device),
                batch["pinyin_tone"].to(device),
                batch["nasal_coda"].to(device),
                batch["radical_idx"].to(device),
                batch["stroke_count"].to(device),
            )
            targets = batch["onyomi_ids"].to(device)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                logits = model(
                    batch["comp_ids"].to(device),
                    batch["comp_mask"].to(device),
                    batch["pinyin_init_idx"].to(device),
                    batch["pinyin_final_idx"].to(device),
                    batch["pinyin_tone"].to(device),
                    batch["nasal_coda"].to(device),
                    batch["radical_idx"].to(device),
                    batch["stroke_count"].to(device),
                )
                targets = batch["onyomi_ids"].to(device)
                test_loss += criterion(logits, targets).item()
                _, predicted = logits.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_acc = test_correct / test_total
        scheduler.step(test_loss)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}: train_loss={train_loss/len(train_loader):.4f}, "
                  f"train_acc={train_correct/train_total:.3f}, "
                  f"test_acc={test_acc:.3f}, lr={optimizer.param_groups[0]['lr']:.6f}")

    model.load_state_dict(best_state)
    return model, best_test_acc


# ── Per-level evaluation ────────────────────────────────────────────
def evaluate_per_level(model, df, tensors, device="cpu"):
    """Evaluate Top-1 and Top-3 accuracy per JLPT level."""
    model.eval()
    model = model.to(device)

    # Get predictions for all test samples
    dataset = KanjiDataset(tensors)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    all_preds = []
    all_confs = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["comp_ids"].to(device),
                batch["comp_mask"].to(device),
                batch["pinyin_init_idx"].to(device),
                batch["pinyin_final_idx"].to(device),
                batch["pinyin_tone"].to(device),
                batch["nasal_coda"].to(device),
                batch["radical_idx"].to(device),
                batch["stroke_count"].to(device),
            )
            probs = torch.softmax(logits, dim=-1)
            top3_probs, top3_indices = probs.topk(3, dim=-1)
            all_preds.extend(top3_indices.cpu().numpy())
            all_confs.extend(top3_probs.cpu().numpy())

    # Map predictions back to dataframe
    test_indices = list(df.index)
    results = []

    for i, idx in enumerate(test_indices):
        row = df.loc[idx]
        true_on = row["onyomi"]
        true_id = row["onyomi_id"]
        level = row["jlpt_level"]

        pred_top1 = id_to_onyomi.get(int(all_preds[i][0]), "?")
        pred_top3 = [id_to_onyomi.get(int(all_preds[i][j]), "?") for j in range(3)]
        correct_top1 = pred_top1 == true_on
        correct_top3 = true_on in pred_top3
        conf_top1 = float(all_confs[i][0])

        results.append({
            "kanji": row["kanji"],
            "true_on": true_on,
            "pred_top1": pred_top1,
            "pred_top3": "|".join(pred_top3),
            "correct_top1": correct_top1,
            "correct_top3": correct_top3,
            "confidence": conf_top1,
            "jlpt_level": level,
        })

    return pd.DataFrame(results)


# ── Run experiments for all variants ────────────────────────────────
device = "cpu"
variants = ["M0", "M1", "M2", "M3"]

print("\n=== Training Model Variants ===\n")
all_results = []

for variant in variants:
    print(f"--- {variant} ---")
    model = create_model(
        variant,
        component_vocab_size=len(component_to_id),
        onyomi_vocab_size=len(onyomi_to_id),
        pinyin_init_vocab_size=len(pinyin_init_to_id),
        pinyin_final_vocab_size=len(pinyin_final_to_id),
        radical_vocab_size=len(radical_to_id),
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    trained_model, test_acc = train_model(
        model, train_loader, test_loader, epochs=50, lr=0.001, device=device
    )
    print(f"  Best test accuracy: {test_acc:.3f}")

    # Per-level evaluation
    results_df = evaluate_per_level(trained_model, test_df, test_tensors, device=device)

    # Save model
    os.makedirs("models/saved", exist_ok=True)
    torch.save(trained_model.state_dict(), f"models/saved/{variant}_best.pt")

    # Per-level accuracy
    print(f"  {'Level':<6} {'Count':<8} {'Top-1':<8} {'Top-3':<8}")
    level_accs = []
    for level in ["N5", "N4", "N3", "N2", "N1", "N1+"]:
        ldf = results_df[results_df["jlpt_level"] == level]
        if len(ldf) == 0:
            continue
        acc1 = ldf["correct_top1"].mean()
        acc3 = ldf["correct_top3"].mean()
        print(f"  {level:<6} {len(ldf):<8} {acc1:<8.1%} {acc3:<8.1%}")
        level_accs.append({
            "variant": variant,
            "level": level,
            "count": len(ldf),
            "top1_acc": acc1,
            "top3_acc": acc3,
        })
    all_results.extend(level_accs)
    print()

# ── Save results ──
results_summary = pd.DataFrame(all_results)
results_summary.to_csv("output/neural_accuracy.csv", index=False)

# Comparison table
print("=== Summary: Baseline vs Neural ===")
baseline_df = pd.read_csv("output/baseline_accuracy.csv")
for level in ["N5", "N4", "N3", "N2", "N1", "N1+"]:
    b = baseline_df[baseline_df["level"] == level]
    if len(b) == 0:
        continue
    b_acc = b.iloc[0]["top1_acc"]
    neural_best = results_summary[results_summary["level"] == level].groupby("variant")["top1_acc"].max().max()
    print(f"  {level}: baseline={b_acc:.1%}, neural_best={neural_best:.1%}, gain=+{neural_best-b_acc:.1%}")

print("\nDone. Model saved to models/saved/")
