"""GPT-2 compatible attention variants for causal language modeling.

Each variant preserves the core projection logic from the ViT implementations
but adds causal masking and conforms to the GPT2Attention interface.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class GPT2LowRankAttention(nn.Module):
    """Low-rank Q/K for GPT-2 causal LM.

    Q = X @ (A^Q @ B^Q), K = X @ (A^K @ B^K) per head.
    V and output projection remain standard.
    """

    def __init__(self, config, low_rank: int, layer_idx: int | None = None):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.low_rank = int(low_rank)
        self.layer_idx = layer_idx

        # Head-wise low-rank factors: A in R^{H, d, r}, B in R^{H, r, d_k}
        self.q_a = nn.Parameter(torch.empty(self.n_head, self.n_embd, self.low_rank))
        self.q_b = nn.Parameter(torch.empty(self.n_head, self.low_rank, self.head_dim))
        self.k_a = nn.Parameter(torch.empty(self.n_head, self.n_embd, self.low_rank))
        self.k_b = nn.Parameter(torch.empty(self.n_head, self.low_rank, self.head_dim))

        self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # Causal mask buffer
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions)).view(
                1, 1, config.n_positions, config.n_positions
            ),
            persistent=False,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        std = 0.02
        for p in (self.q_a, self.q_b, self.k_a, self.k_b):
            nn.init.normal_(p, mean=0.0, std=std)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=std)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)

    def _project(self, x: torch.Tensor, a: nn.Parameter, b: nn.Parameter) -> torch.Tensor:
        """X @ (A @ B) via einsum: [B,T,d] x [H,d,r] x [H,r,d_k] -> [B,H,T,d_k]"""
        mid = torch.einsum("btd,hdr->bhtr", x, a)
        out = torch.einsum("bhtr,hrd->bhtd", mid, b)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        B, T, C = hidden_states.size()

        q = self._project(hidden_states, self.q_a, self.q_b)
        k = self._project(hidden_states, self.k_a, self.k_b)
        v = self.v_proj(hidden_states)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        causal_mask = self.bias[:, :, :T, :T]
        att = att.masked_fill(causal_mask == 0, float("-inf"))
        if attention_mask is not None:
            att = att + attention_mask

        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))

        return (y, att if output_attentions else None)


class GPT2FullySharedAttention(nn.Module):
    """Reformer-style fully shared Q=K for GPT-2 causal LM."""

    def __init__(self, config, layer_idx: int | None = None):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.layer_idx = layer_idx

        self.query_key = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions)).view(
                1, 1, config.n_positions, config.n_positions
            ),
            persistent=False,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.query_key.weight, mean=0.0, std=std)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=std)
        for m in (self.query_key, self.v_proj, self.o_proj):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        B, T, C = hidden_states.size()

        q = self.query_key(hidden_states).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.query_key(hidden_states).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        causal_mask = self.bias[:, :, :T, :T]
        att = att.masked_fill(causal_mask == 0, float("-inf"))
        if attention_mask is not None:
            att = att + attention_mask

        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))

        return (y, att if output_attentions else None)


class GPT2UVLatentAttention(nn.Module):
    """BMB-UV latent attention for GPT-2 causal LM.

    Shared latent basis Z = X @ B, then head-specific UV factors:
        Q = einsum(Z, U_factor), K = einsum(Z, V_factor)
    """

    def __init__(
        self,
        config,
        latent_rank: int,
        latent_factor_rank: int | None = None,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.latent_rank = int(latent_rank)
        self.latent_factor_rank = int(latent_factor_rank if latent_factor_rank is not None else latent_rank)
        self.layer_idx = layer_idx

        self.basis = nn.Linear(self.n_embd, self.latent_rank, bias=False)
        self.u_factor = nn.Parameter(torch.empty(self.n_head, self.latent_rank, self.latent_factor_rank))
        self.v_factor = nn.Parameter(torch.empty(self.n_head, self.latent_rank, self.latent_factor_rank))
        self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions)).view(
                1, 1, config.n_positions, config.n_positions
            ),
            persistent=False,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.basis.weight, mean=0.0, std=std)
        nn.init.normal_(self.u_factor, mean=0.0, std=std)
        nn.init.normal_(self.v_factor, mean=0.0, std=std)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=std)
        for m in (self.v_proj, self.o_proj):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        B, T, C = hidden_states.size()

        latent = self.basis(hidden_states)  # [B, T, r]
        q = torch.einsum("btr,hrs->bhts", latent, self.u_factor)
        k = torch.einsum("btr,hrs->bhts", latent, self.v_factor)
        v = self.v_proj(hidden_states).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.latent_factor_rank))
        causal_mask = self.bias[:, :, :T, :T]
        att = att.masked_fill(causal_mask == 0, float("-inf"))
        if attention_mask is not None:
            att = att + attention_mask

        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))

        return (y, att if output_attentions else None)


class GPT2BBTAttention(nn.Module):
    """BBT attention for GPT-2 causal LM.

    A shared latent basis forms identical query/key coordinates for all heads.
    Head diversity is carried only by the value pathway.
    """

    def __init__(self, config, latent_rank: int, layer_idx: int | None = None):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.latent_rank = int(latent_rank)
        self.layer_idx = layer_idx

        self.basis = nn.Linear(self.n_embd, self.latent_rank, bias=False)
        self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions)).view(
                1, 1, config.n_positions, config.n_positions
            ),
            persistent=False,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.basis.weight, mean=0.0, std=std)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=std)
        for m in (self.v_proj, self.o_proj):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        B, T, C = hidden_states.size()

        latent = self.basis(hidden_states)  # [B, T, r]
        q = latent.unsqueeze(1).expand(-1, self.n_head, -1, -1)
        k = latent.unsqueeze(1).expand(-1, self.n_head, -1, -1)
        v = self.v_proj(hidden_states).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.latent_rank))
        causal_mask = self.bias[:, :, :T, :T]
        att = att.masked_fill(causal_mask == 0, float("-inf"))
        if attention_mask is not None:
            att = att + attention_mask

        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))

        return (y, att if output_attentions else None)


class GPT2SymmetricLatentAttention(nn.Module):
    """BMB attention for GPT-2 causal LM.

    Heads share a latent basis but use head-specific bilinear interaction
    operators inside that basis.
    """

    def __init__(self, config, latent_rank: int, layer_idx: int | None = None):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.latent_rank = int(latent_rank)
        self.layer_idx = layer_idx

        self.basis = nn.Linear(self.n_embd, self.latent_rank, bias=False)
        self.core = nn.Parameter(torch.empty(self.latent_rank, self.latent_rank))
        self.head_residual = nn.Parameter(
            torch.empty(self.n_head, self.latent_rank, self.latent_rank)
        )
        self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions)).view(
                1, 1, config.n_positions, config.n_positions
            ),
            persistent=False,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.basis.weight, mean=0.0, std=std)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=std)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)
        nn.init.normal_(self.core, mean=0.0, std=std)
        nn.init.normal_(self.head_residual, mean=0.0, std=std / 10.0)

    def latent_core(self) -> torch.Tensor:
        return 0.5 * (self.core + self.core.transpose(0, 1))

    def head_matrices(self) -> torch.Tensor:
        centered = self.head_residual - self.head_residual.mean(dim=0, keepdim=True)
        return self.latent_core().unsqueeze(0) / float(self.n_head) + centered

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        B, T, C = hidden_states.size()

        latent = self.basis(hidden_states)  # [B, T, r]
        head_mats = self.head_matrices()  # [H, r, r]
        att = torch.einsum("btr,hrs,bus->bhtu", latent, head_mats, latent)
        att = att * (1.0 / math.sqrt(self.latent_rank))
        causal_mask = self.bias[:, :, :T, :T]
        att = att.masked_fill(causal_mask == 0, float("-inf"))
        if attention_mask is not None:
            att = att + attention_mask

        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        v = self.v_proj(hidden_states).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))

        return (y, att if output_attentions else None)


class GPT2PartialSharedAttention(nn.Module):
    """Partial shared QK for GPT-2 causal LM.

    W^Q_h = [W^{share} | W^{Q,priv}_h], W^K_h = [W^{share} | W^{K,priv}_h]
    where W^{share} is truly shared across heads.
    """

    def __init__(self, config, shared_qk_dim: int, layer_idx: int | None = None):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.shared_qk_dim = int(shared_qk_dim)
        self.private_qk_dim = self.head_dim - self.shared_qk_dim
        if self.private_qk_dim < 0:
            raise ValueError(f"shared_qk_dim ({shared_qk_dim}) > head_dim ({self.head_dim})")
        self.layer_idx = layer_idx

        self.share = nn.Linear(self.n_embd, self.shared_qk_dim, bias=True)
        if self.private_qk_dim > 0:
            self.query_priv = nn.Linear(self.n_embd, self.n_head * self.private_qk_dim, bias=True)
            self.key_priv = nn.Linear(self.n_embd, self.n_head * self.private_qk_dim, bias=True)
        else:
            self.query_priv = None
            self.key_priv = None
        self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions)).view(
                1, 1, config.n_positions, config.n_positions
            ),
            persistent=False,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        std = 0.02
        modules = [self.share, self.v_proj, self.o_proj]
        if self.query_priv is not None:
            modules.extend([self.query_priv, self.key_priv])
        for m in modules:
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _project(self, x: torch.Tensor, proj: nn.Linear, width: int) -> torch.Tensor:
        out = proj(x)  # [B, T, H*width]
        B, T, _ = out.shape
        out = out.view(B, T, self.n_head, width)
        return out.transpose(1, 2)  # [B, H, T, width]

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        B, T, C = hidden_states.size()

        share = self.share(hidden_states)  # [B, T, m]
        share = share.unsqueeze(1).expand(-1, self.n_head, -1, -1)

        if self.private_qk_dim > 0:
            q_priv = self._project(hidden_states, self.query_priv, self.private_qk_dim)
            k_priv = self._project(hidden_states, self.key_priv, self.private_qk_dim)
            q = torch.cat([share, q_priv], dim=-1)
            k = torch.cat([share, k_priv], dim=-1)
        else:
            q = share
            k = share
        v = self.v_proj(hidden_states).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        causal_mask = self.bias[:, :, :T, :T]
        att = att.masked_fill(causal_mask == 0, float("-inf"))
        if attention_mask is not None:
            att = att + attention_mask

        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))

        return (y, att if output_attentions else None)


def replace_gpt2_attention(model: nn.Module, variant: str, **kwargs: Any) -> None:
    """In-place replace attention layers in a GPT-2 model."""
    for i, block in enumerate(model.transformer.h):
        if variant == "lowrank":
            block.attn = GPT2LowRankAttention(model.config, low_rank=kwargs["rank"], layer_idx=i)
        elif variant == "fullyshared":
            block.attn = GPT2FullySharedAttention(model.config, layer_idx=i)
        elif variant == "bbt":
            block.attn = GPT2BBTAttention(model.config, latent_rank=kwargs["rank"], layer_idx=i)
        elif variant == "bmb":
            block.attn = GPT2SymmetricLatentAttention(
                model.config, latent_rank=kwargs["rank"], layer_idx=i
            )
        elif variant == "bmbuv":
            block.attn = GPT2UVLatentAttention(
                model.config,
                latent_rank=kwargs["rank"],
                latent_factor_rank=kwargs.get("factor_rank"),
                layer_idx=i,
            )
        elif variant == "partialshared":
            block.attn = GPT2PartialSharedAttention(
                model.config, shared_qk_dim=kwargs["shared_dim"], layer_idx=i
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")
