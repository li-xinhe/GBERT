import torch
import torch.nn as nn
from transformers import AutoModel


class MacroSubNet(nn.Module):
    """Single macro variable: scalar -> per-class logit contribution (kept for interpretability)."""
    def __init__(self, num_classes: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)  # (batch, num_classes)


class MacroTokenEncoder(nn.Module):
    """Encode each macro scalar into a token embedding."""
    def __init__(self, d_model: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):  # x: (batch, 1)
        return self.net(x)  # (batch, d_model)


class CausalInterpretableModel(nn.Module):
    """
    Same inputs/outputs as your current model, but with stronger interaction between:
    text <-> (macro, country, year) via:
      (1) a structural-context Transformer over macro/country/year tokens
      (2) FiLM modulation of CLS embedding
      (3) an explicit interaction head (still decomposable)
    """
    def __init__(
        self,
        model_name="bert-base-multilingual-cased",
        num_classes=56,
        num_macro=6,
        num_countries=50,
        num_years=100,
        text_hidden_dim=256,
        macro_hidden_dim=16,
        country_emb_dim=16,
        year_emb_dim=16,
        ctx_dim=128,
        ctx_depth=2,
        ctx_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_macro = num_macro

        # ----- Text encoder -----
        self.bert = AutoModel.from_pretrained(model_name)
        bert_dim = self.bert.config.hidden_size

        # Base text head (kept)
        self.text_proj = nn.Sequential(
            nn.Linear(bert_dim, text_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(text_hidden_dim, num_classes),
        )

        # ----- Additive macro heads (kept) -----
        self.macro_nets = nn.ModuleList([
            MacroSubNet(num_classes=num_classes, hidden_dim=macro_hidden_dim)
            for _ in range(num_macro)
        ])

        # ----- Country/Year embeddings (kept) -----
        self.country_emb = nn.Embedding(num_countries, country_emb_dim)
        self.country_head = nn.Linear(country_emb_dim, num_classes)

        self.year_emb = nn.Embedding(num_years, year_emb_dim)
        self.year_head = nn.Linear(year_emb_dim, num_classes)

        # ----- Structural context encoder (NEW) -----
        # macro scalars -> ctx tokens
        self.macro_token = MacroTokenEncoder(d_model=ctx_dim, hidden=max(64, ctx_dim // 2), dropout=dropout)

        # country/year -> ctx tokens
        self.country_to_ctx = nn.Sequential(
            nn.Linear(country_emb_dim, ctx_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.year_to_ctx = nn.Sequential(
            nn.Linear(year_emb_dim, ctx_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=ctx_dim,
            nhead=ctx_heads,
            dim_feedforward=ctx_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.ctx_encoder = nn.TransformerEncoder(enc_layer, num_layers=ctx_depth)
        self.ctx_norm = nn.LayerNorm(ctx_dim)

        # ----- FiLM modulation of CLS by context (NEW) -----
        # gamma/beta in bert_dim space
        self.ctx_to_gamma_beta = nn.Sequential(
            nn.Linear(ctx_dim, bert_dim * 2),
            nn.Tanh(),  # keeps modulation bounded
        )

        # ----- Explicit interaction head (NEW) -----
        # produces per-class logit adjustments due to text-structure interaction
        self.ctx_to_bert = nn.Linear(ctx_dim, bert_dim)
        self.interaction_head = nn.Sequential(
            nn.Linear(bert_dim, text_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_hidden_dim, num_classes),
        )

    def _encode_context(self, macro, country_id, year_id):
        """
        macro: (batch, num_macro)
        returns:
          ctx: (batch, ctx_dim)
          country_vec/year_vec: embeddings for optional components output
        """
        country_vec = self.country_emb(country_id)  # (batch, country_emb_dim)
        year_vec = self.year_emb(year_id)          # (batch, year_emb_dim)

        # Build tokens: [macro_1 ... macro_M, country_token, year_token]
        macro_tokens = []
        for j in range(self.num_macro):
            xj = macro[:, j:j+1]               # (batch, 1)
            macro_tokens.append(self.macro_token(xj))  # (batch, ctx_dim)
        macro_tokens = torch.stack(macro_tokens, dim=1)  # (batch, M, ctx_dim)

        c_tok = self.country_to_ctx(country_vec).unsqueeze(1)  # (batch, 1, ctx_dim)
        y_tok = self.year_to_ctx(year_vec).unsqueeze(1)        # (batch, 1, ctx_dim)
        tokens = torch.cat([macro_tokens, c_tok, y_tok], dim=1)  # (batch, M+2, ctx_dim)

        # Let tokens interact
        tokens = self.ctx_encoder(tokens)  # (batch, M+2, ctx_dim)
        ctx = tokens.mean(dim=1)           # (batch, ctx_dim)
        ctx = self.ctx_norm(ctx)
        return ctx, country_vec, year_vec

    def forward(self, input_ids, attention_mask, macro, country_id, year_id, output_components=False):
        """
        Inputs/outputs kept the same:
          - If output_components=False: return logits
          - If True: return (logits, dict)
        """
        # ----- Context (macro+country+year) -----
        ctx, country_vec, year_vec = self._encode_context(macro, country_id, year_id)

        # ----- Text encoding -----
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = out.last_hidden_state[:, 0, :]  # (batch, bert_dim)

        # ----- FiLM modulation (text conditioned on context) -----
        gamma_beta = self.ctx_to_gamma_beta(ctx)          # (batch, 2*bert_dim)
        gamma, beta = gamma_beta.chunk(2, dim=-1)         # each (batch, bert_dim)
        cls_film = cls_emb * (1.0 + gamma) + beta         # (batch, bert_dim)

        # Text logits (now structurally conditioned)
        logits_text = self.text_proj(cls_film)            # (batch, num_classes)

        # ----- Additive macro contributions (kept) -----
        macro_contribs = []
        for j, net in enumerate(self.macro_nets):
            xj = macro[:, j:j+1]        # (batch, 1)
            macro_contribs.append(net(xj))
        logits_macro = torch.stack(macro_contribs, dim=0).sum(dim=0)  # (batch, num_classes)

        # ----- Country/year main effects (kept) -----
        logits_country = self.country_head(country_vec)   # (batch, num_classes)
        logits_year = self.year_head(year_vec)            # (batch, num_classes)

        # ----- Explicit interaction term (NEW, but still decomposable) -----
        ctx_in_bert = self.ctx_to_bert(ctx)               # (batch, bert_dim)
        interaction_feat = cls_emb * ctx_in_bert 
        logits_interaction = self.interaction_head(interaction_feat)  # (batch, num_classes)

        # ----- Total logits -----
        logits = logits_text + logits_interaction + logits_macro + logits_country + logits_year #

        if output_components:
            return logits, {
                "logits_text": logits_text,
                "logits_macro_each": macro_contribs,
                "logits_macro_sum": logits_macro,
                "logits_country": logits_country,
                "logits_year": logits_year,
                "logits_interaction": logits_interaction,  # extra, backward-compatible
            }
        return logits
