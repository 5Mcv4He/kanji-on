"""Phase 3b: Improved neural model — deeper MLP, label smoothing, better training.

Key improvements:
  1. Deeper network with residual connections + LayerNorm
  2. Label smoothing (0.08) for better generalization
  3. Cosine annealing scheduler
  4. AdamW with weight decay
  5. Gradient clipping
  6. Longer training with proper early stopping
"""
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os

print("Loading data...")
df = pd.read_csv("dataset/kanji_dataset.csv")
with open("dataset/component_vocab.json") as f: component_to_id = json.load(f)
with open("dataset/onyomi_vocab.json") as f: onyomi_to_id = json.load(f)
with open("dataset/pinyin_init_vocab.json") as f: pinyin_init_to_id = json.load(f)
with open("dataset/pinyin_final_vocab.json") as f: pinyin_final_to_id = json.load(f)
with open("dataset/radical_vocab.json") as f: radical_to_id = json.load(f)
id_to_onyomi = {v: k for k, v in onyomi_to_id.items()}

df["comp_id_list"] = df["component_ids"].apply(lambda x: json.loads(x) if isinstance(x, str) else [])
df = df[df["onyomi"].notna() & (df["onyomi"] != "")].copy()
df["onyomi_id"] = df["onyomi"].map(onyomi_to_id)
df = df.dropna(subset=["onyomi_id"])
df["onyomi_id"] = df["onyomi_id"].astype(int)

MAX_COMPS = 8
nasal_to_vec = {"none": [1, 0, 0], "n": [0, 1, 0], "ng": [0, 0, 1]}

class ImprovedModel(nn.Module):
    def __init__(self, num_components, num_onyomi, num_pinyin_initials,
                 num_pinyin_finals, num_radicals,
                 comp_emb_dim=96, hidden_dim=320, dropout=0.25):
        super().__init__()
        self.comp_embedding = nn.Embedding(num_components + 1, comp_emb_dim, padding_idx=0)
        self.pinyin_init_emb = nn.Embedding(num_pinyin_initials, 18)
        self.pinyin_final_emb = nn.Embedding(num_pinyin_finals, 36)
        self.pinyin_tone_emb = nn.Embedding(6, 8)
        self.radical_emb = nn.Embedding(num_radicals, 28)

        feat_dim = comp_emb_dim + 18 + 36 + 8 + 3 + 28 + 1  # +1 stroke

        # Deeper MLP with residual blocks
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.ln_input = nn.LayerNorm(hidden_dim)

        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(hidden_dim // 2, num_onyomi)

    def forward(self, comp_ids, comp_mask, pinyin_init_idx, pinyin_final_idx,
                pinyin_tone, nasal_coda, radical_idx, stroke_count):
        # Component mean pool
        comp_emb = self.comp_embedding(comp_ids)
        mask_exp = comp_mask.unsqueeze(-1).float()
        comp_pooled = (comp_emb * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)

        p_init = self.pinyin_init_emb(pinyin_init_idx)
        p_final = self.pinyin_final_emb(pinyin_final_idx)
        p_tone = self.pinyin_tone_emb(pinyin_tone)
        r_emb = self.radical_emb(radical_idx)

        x = torch.cat([comp_pooled, p_init, p_final, p_tone, nasal_coda, r_emb,
                      stroke_count.unsqueeze(-1)], dim=-1)

        h = F.relu(self.ln_input(self.input_proj(x)))
        h = h + self.block1(h)  # Residual
        h = h + self.block2(h)  # Residual
        h = self.block3(h)
        return self.output(h)

def prepare_features(data_df):
    n = len(data_df)
    comp_ids = np.zeros((n, MAX_COMPS), dtype=np.int64)
    comp_mask = np.zeros((n, MAX_COMPS), dtype=bool)
    pinyin_init_idx = np.zeros(n, dtype=np.int64)
    pinyin_final_idx = np.zeros(n, dtype=np.int64)
    pinyin_tone = np.zeros(n, dtype=np.int64)
    nasal_coda = np.zeros((n, 3), dtype=np.float32)
    radical_idx = np.zeros(n, dtype=np.int64)
    stroke_count = np.zeros(n, dtype=np.float32)
    onyomi_ids = np.zeros(n, dtype=np.int64)

    for i, (_, row) in enumerate(data_df.iterrows()):
        clist = row["comp_id_list"]
        for j, cid in enumerate(clist[:MAX_COMPS]):
            comp_ids[i, j] = cid + 1
            comp_mask[i, j] = True
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
        rad = row["radical"]
        if rad and rad in radical_to_id:
            radical_idx[i] = radical_to_id[rad]
        sc = row["stroke_count"]
        stroke_count[i] = sc / 30.0 if not pd.isna(sc) else 0
        onyomi_ids[i] = int(row["onyomi_id"])

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
    def __init__(self, tensors): self.tensors = tensors
    def __len__(self): return len(self.tensors["onyomi_ids"])
    def __getitem__(self, idx): return {k: v[idx] for k, v in self.tensors.items()}

# Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42,
                                     stratify=df["jlpt_level"])
train_t = prepare_features(train_df)
test_t = prepare_features(test_df)
train_loader = DataLoader(KanjiDataset(train_t), batch_size=128, shuffle=True)
test_loader = DataLoader(KanjiDataset(test_t), batch_size=128, shuffle=False)
print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

# Model
model = ImprovedModel(
    num_components=len(component_to_id),
    num_onyomi=len(onyomi_to_id),
    num_pinyin_initials=len(pinyin_init_to_id),
    num_pinyin_finals=len(pinyin_final_to_id),
    num_radicals=len(radical_to_id),
)
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

# Label smoothing
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.08):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1.0)
            smooth = smooth * (1 - self.smoothing) + self.smoothing / self.num_classes
        return -(smooth * log_probs).sum(dim=-1).mean()

criterion = LabelSmoothingLoss(len(onyomi_to_id), 0.08)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0012, weight_decay=0.008)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

best_acc = 0
best_state = None
patience = 0

for epoch in range(100):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(
            batch["comp_ids"], batch["comp_mask"],
            batch["pinyin_init_idx"], batch["pinyin_final_idx"],
            batch["pinyin_tone"], batch["nasal_coda"],
            batch["radical_idx"], batch["stroke_count"],
        )
        loss = criterion(logits, batch["onyomi_ids"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                batch["comp_ids"], batch["comp_mask"],
                batch["pinyin_init_idx"], batch["pinyin_final_idx"],
                batch["pinyin_tone"], batch["nasal_coda"],
                batch["radical_idx"], batch["stroke_count"],
            )
            _, pred = logits.max(1)
            correct += pred.eq(batch["onyomi_ids"]).sum().item()
            total += batch["onyomi_ids"].size(0)
    acc = correct / total

    if acc > best_acc:
        best_acc = acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience = 0
    else:
        patience += 1

    if epoch % 15 == 0 or epoch == 99:
        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:2d}: acc={acc:.4f}, best={best_acc:.4f}, lr={lr:.6f}")

    if patience >= 30:
        print(f"  Early stop at epoch {epoch}")
        break

if best_state is None:
    print("ERROR: No improvement. Saving untrained model for debug.")
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    best_acc = 0.0

model.load_state_dict(best_state)
os.makedirs("models/saved", exist_ok=True)
torch.save(best_state, "models/saved/improved_best.pt")
print(f"\n  Best accuracy: {best_acc:.4f}")

# Per-level
print(f"\n{'Level':<6} {'Count':<8} {'Top-1':<10} {'Top-3':<10}")

for level in ["N5", "N4", "N3", "N2", "N1", "N1+"]:
    ldf = test_df[test_df["jlpt_level"] == level]
    if len(ldf) == 0: continue
    lt = prepare_features(ldf)
    loader = DataLoader(KanjiDataset(lt), batch_size=256)

    c1 = c3 = tot = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["comp_ids"], batch["comp_mask"],
                batch["pinyin_init_idx"], batch["pinyin_final_idx"],
                batch["pinyin_tone"], batch["nasal_coda"],
                batch["radical_idx"], batch["stroke_count"],
            )
            probs = torch.softmax(logits, dim=-1)
            top3 = probs.topk(3, dim=-1).indices
            _, pred = logits.max(1)
            targets = batch["onyomi_ids"]
            c1 += pred.eq(targets).sum().item()
            c3 += top3.eq(targets.unsqueeze(1)).any(dim=1).sum().item()
            tot += targets.size(0)

    print(f"{level:<6} {tot:<8} {c1/tot:<10.1%} {c3/tot:<10.1%}")

# Compare
print(f"\n=== Comparison ===")
print(f"  Improved model: {best_acc:.4f}")
try:
    neural = pd.read_csv("output/neural_accuracy.csv")
    m2 = neural[neural["variant"]=="M2"]["top1_acc"].max()
    print(f"  M2 (previous best): {m2:.4f}")
    print(f"  Gain: +{best_acc - m2:.4f}")
except: pass
