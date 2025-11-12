import torch
import torch.nn as nn
import torch.nn.functional as F

class SyntaxAwareDecoder(nn.Module):
    def __init__(self, d_enc, d_model, vocab_size, num_rel=7):
        super().__init__()
        self.gru_alpha = nn.GRUCell(d_model, d_model)
        self.gru_beta  = nn.GRUCell(d_model, d_model)

        self.W_enc = nn.Linear(d_enc, d_model)   # project Qwen feats
        self.W_co  = nn.Linear(d_model, d_model)
        self.W_cov = nn.Linear(d_model, d_model)
        self.W_score = nn.Linear(d_model, 1)

        self.symbol_head = nn.Linear(d_model, vocab_size + 2)  # Σ + {E, ε}
        self.rel_head    = nn.Linear(d_model, num_rel)

    def attend(self, enc_tokens, c_o, cov_vec):
        # enc_tokens: [L, d_enc]
        e = self.W_enc(enc_tokens)                 # [L, d_model]
        co = self.W_co(c_o).unsqueeze(0)           # [1, d_model] -> [L, d_model]
        co = co.expand_as(e)
        cov = self.W_cov(cov_vec).unsqueeze(0).expand_as(e)

        scores = self.W_score(torch.tanh(e + co + cov)).squeeze(-1)  # [L]
        alpha = F.softmax(scores, dim=-1)
        ctx = torch.matmul(alpha.unsqueeze(0), enc_tokens).squeeze(0)  # [d_enc]
        return alpha, ctx

    def forward_step(self, enc_tokens, c_prev, partner_embed, cov_vec):
        # 1) history
        c_o = self.gru_alpha(partner_embed, c_prev)
        # 2) syntax-aware attention
        alpha, ctx = self.attend(enc_tokens, c_o, cov_vec)
        # 3) fuse context
        c_beta = self.gru_beta(ctx, c_o)
        # 4) predict
        symbol_logits = self.symbol_head(c_beta)
        rel_logits    = self.rel_head(c_beta)
        return symbol_logits, rel_logits, c_beta, alpha