"""Phase 3: Small neural network for kanji on'yomi prediction."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class KanjiOnyomiModel(nn.Module):
    """Predict on'yomi from kanji components + pinyin features.

    Architecture variants:
      M0: Components only
      M1: Components + Pinyin
      M2: Components + Pinyin + Radical
      M3: Components + Pinyin + Radical + StrokeCount
    """

    def __init__(self,
                 num_components: int,
                 num_onyomi: int,
                 num_pinyin_initials: int,
                 num_pinyin_finals: int,
                 num_radicals: int,
                 comp_emb_dim: int = 64,
                 pinyin_init_emb_dim: int = 8,
                 pinyin_final_emb_dim: int = 16,
                 radical_emb_dim: int = 16,
                 hidden_dims: tuple = (128, 64),
                 dropout: float = 0.3,
                 use_pinyin: bool = True,
                 use_radical: bool = True,
                 use_stroke: bool = True):
        super().__init__()

        self.use_pinyin = use_pinyin
        self.use_radical = use_radical
        self.use_stroke = use_stroke

        # Component embedding (bag-of-components → dense vector)
        self.comp_embedding = nn.Embedding(num_components, comp_emb_dim, padding_idx=0)

        # Calculate total input dim
        input_dim = comp_emb_dim

        if use_pinyin:
            self.pinyin_init_emb = nn.Embedding(num_pinyin_initials, pinyin_init_emb_dim)
            self.pinyin_final_emb = nn.Embedding(num_pinyin_finals, pinyin_final_emb_dim)
            self.pinyin_tone_emb = nn.Embedding(6, 4)  # tones 0-5
            input_dim += pinyin_init_emb_dim + pinyin_final_emb_dim + 4 + 3  # +3 for nasal coda OHE

        if use_radical:
            self.radical_emb = nn.Embedding(num_radicals, radical_emb_dim)
            input_dim += radical_emb_dim

        if use_stroke:
            input_dim += 1  # stroke_count normalized

        # MLP head
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_onyomi))
        self.mlp = nn.Sequential(*layers)

    def forward(self, comp_ids, comp_mask, pinyin_init_idx, pinyin_final_idx,
                pinyin_tone, nasal_coda, radical_idx, stroke_count):
        """Forward pass.

        Args:
            comp_ids: (B, max_comps) component token indices, 0 = padding
            comp_mask: (B, max_comps) boolean mask
            pinyin_init_idx: (B,) initial consonant index
            pinyin_final_idx: (B,) final index
            pinyin_tone: (B,) tone 0-5
            nasal_coda: (B, 3) one-hot [none, n, ng]
            radical_idx: (B,) radical index
            stroke_count: (B,) normalized stroke count
        """
        # Component embeddings → mean pool
        comp_emb = self.comp_embedding(comp_ids)  # (B, max_comps, comp_emb_dim)
        comp_mask_expanded = comp_mask.unsqueeze(-1).float()  # (B, max_comps, 1)
        comp_emb = comp_emb * comp_mask_expanded
        comp_sum = comp_emb.sum(dim=1)  # (B, comp_emb_dim)
        comp_count = comp_mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
        comp_pooled = comp_sum / comp_count  # (B, comp_emb_dim)

        features = [comp_pooled]

        if self.use_pinyin:
            p_init = self.pinyin_init_emb(pinyin_init_idx)
            p_final = self.pinyin_final_emb(pinyin_final_idx)
            p_tone = self.pinyin_tone_emb(pinyin_tone)
            features.extend([p_init, p_final, p_tone, nasal_coda])

        if self.use_radical:
            r_emb = self.radical_emb(radical_idx)
            features.append(r_emb)

        if self.use_stroke:
            features.append(stroke_count.unsqueeze(-1))

        x = torch.cat(features, dim=-1)
        return self.mlp(x)


class JointOnKunModel(nn.Module):
    """Multi-task model: on'yomi classification + kun'yomi type classification.
    Shared encoder with separate output heads.
    """
    def __init__(self, num_components, num_onyomi, num_kun_types,
                 num_pinyin_initials, num_pinyin_finals, num_radicals,
                 comp_emb_dim=64, pinyin_init_emb_dim=8, pinyin_final_emb_dim=16,
                 radical_emb_dim=16, hidden_dims=(128, 64), dropout=0.3):
        super().__init__()
        self.comp_embedding = nn.Embedding(num_components + 1, comp_emb_dim, padding_idx=0)
        self.pinyin_init_emb = nn.Embedding(num_pinyin_initials, pinyin_init_emb_dim)
        self.pinyin_final_emb = nn.Embedding(num_pinyin_finals, pinyin_final_emb_dim)
        self.pinyin_tone_emb = nn.Embedding(6, 4)
        self.radical_emb = nn.Embedding(num_radicals, radical_emb_dim)

        input_dim = comp_emb_dim + pinyin_init_emb_dim + pinyin_final_emb_dim + 4 + 3 + radical_emb_dim + 1

        shared_layers = []
        prev = input_dim
        for h_dim in hidden_dims:
            shared_layers.append(nn.Linear(prev, h_dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout))
            prev = h_dim
        self.shared = nn.Sequential(*shared_layers)
        self.onyomi_head = nn.Linear(prev, num_onyomi)
        self.kun_head = nn.Linear(prev, num_kun_types)

    def forward(self, comp_ids, comp_mask, pinyin_init_idx, pinyin_final_idx,
                pinyin_tone, nasal_coda, radical_idx, stroke_count):
        comp_emb = self.comp_embedding(comp_ids)
        comp_mask_exp = comp_mask.unsqueeze(-1).float()
        comp_emb = comp_emb * comp_mask_exp
        comp_pooled = comp_emb.sum(dim=1) / comp_mask_exp.sum(dim=1).clamp(min=1)

        p_init = self.pinyin_init_emb(pinyin_init_idx)
        p_final = self.pinyin_final_emb(pinyin_final_idx)
        p_tone = self.pinyin_tone_emb(pinyin_tone)
        r_emb = self.radical_emb(radical_idx)

        x = torch.cat([comp_pooled, p_init, p_final, p_tone, nasal_coda, r_emb, stroke_count.unsqueeze(-1)], dim=-1)
        shared = self.shared(x)
        return self.onyomi_head(shared), self.kun_head(shared)


def create_model(variant: str, component_vocab_size: int, onyomi_vocab_size: int,
                 pinyin_init_vocab_size: int, pinyin_final_vocab_size: int,
                 radical_vocab_size: int, **kwargs) -> KanjiOnyomiModel:
    """Factory for model variants M0-M3."""
    configs = {
        "M0": {"use_pinyin": False, "use_radical": False, "use_stroke": False},
        "M1": {"use_pinyin": True, "use_radical": False, "use_stroke": False},
        "M2": {"use_pinyin": True, "use_radical": True, "use_stroke": False},
        "M3": {"use_pinyin": True, "use_radical": True, "use_stroke": True},
    }
    cfg = configs[variant]
    return KanjiOnyomiModel(
        num_components=component_vocab_size + 1,  # +1 for padding idx 0
        num_onyomi=onyomi_vocab_size,
        num_pinyin_initials=pinyin_init_vocab_size,
        num_pinyin_finals=pinyin_final_vocab_size,
        num_radicals=radical_vocab_size,
        **cfg,
        **kwargs
    )
