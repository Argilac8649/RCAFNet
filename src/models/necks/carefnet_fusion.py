import math

import torch
import torch.nn as nn

try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_


def _maybe_to_cpu(x: torch.Tensor, to_cpu: bool = False) -> torch.Tensor:
    return x.cpu() if to_cpu else x


def _debug_feature_map(x: torch.Tensor, to_cpu: bool = False) -> torch.Tensor:
    """将特征张量转换为空间响应图。

    这里使用通道维绝对值均值作为可视化响应，避免依赖具体模块结构。
    默认保留在原设备上，仅在显式要求时搬到 CPU。
    """
    if x.ndim != 4:
        raise ValueError(f"Expected NCHW feature map, got {tuple(x.shape)}.")
    value = x.detach().abs().mean(dim=1, keepdim=True).float()
    return _maybe_to_cpu(value, to_cpu)


def _debug_token_map(
    x: torch.Tensor,
    height: int,
    width: int,
    to_cpu: bool = False,
) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected [B, N, C] token map, got {tuple(x.shape)}.")
    batch, tokens, _ = x.shape
    if tokens != int(height) * int(width):
        raise ValueError(
            f"Expected N=H*W for token debug map, got N={tokens}, "
            f"H={height}, W={width}."
        )
    value = x.detach().abs().mean(dim=-1).reshape(batch, 1, height, width).float()
    return _maybe_to_cpu(value, to_cpu)


def _debug_detach(x: torch.Tensor, to_cpu: bool = False) -> torch.Tensor:
    return _maybe_to_cpu(x.detach().float(), to_cpu)


# Fusion

def _normalize_attention_type(attention_type):
    attention_type = str(attention_type).strip().lower()
    allowed = {"efficient"}
    if attention_type not in allowed:
        raise ValueError(
            f"Unknown attention_type='{attention_type}'. "
            "Expected 'efficient'."
        )
    return attention_type


def _check_token_pair(x1, x2, dim: int, module_name: str):
    if x1.ndim != 3 or x2.ndim != 3:
        raise ValueError(
            f"{module_name} expects [B, N, C] tokens, got "
            f"x1={tuple(x1.shape)}, x2={tuple(x2.shape)}."
        )
    if x1.shape != x2.shape:
        raise ValueError(
            f"{module_name} expects matched token shapes, got "
            f"x1={tuple(x1.shape)}, x2={tuple(x2.shape)}."
        )
    if x1.shape[-1] != int(dim):
        raise ValueError(
            f"{module_name} channel mismatch: got C={x1.shape[-1]}, "
            f"expected {int(dim)}."
        )
    return x1.shape


def _check_feature_pair(x1, x2, channels: int, module_name: str):
    if x1.ndim != 4 or x2.ndim != 4:
        raise ValueError(
            f"{module_name} expects NCHW inputs, got "
            f"x1={tuple(x1.shape)}, x2={tuple(x2.shape)}."
        )
    if x1.shape != x2.shape:
        raise ValueError(
            f"{module_name} expects matched feature shapes, got "
            f"x1={tuple(x1.shape)}, x2={tuple(x2.shape)}."
        )
    if x1.shape[1] != int(channels):
        raise ValueError(
            f"{module_name} channel mismatch: got C={x1.shape[1]}, "
            f"expected {int(channels)}."
        )
    return x1.shape


class EfficientTokenCrossAttention(nn.Module):
    """
    Efficient cross-modal attention.

    This keeps the original efficient attention design:

        context = softmax(K^T V)
        output  = Q context

    Input:
        x1, x2: [B, N, C]

    Output:
        out1: x1 reads the context of x2
        out2: x2 reads the context of x1
    """

    def __init__(
        self,
        dim,
        num_heads=1,
        qkv_bias=False,
        attn_drop=0.0,
    ):
        super(EfficientTokenCrossAttention, self).__init__()
        self.dim = int(dim)
        if num_heads is None:
            num_heads = 1
        self.num_heads = int(num_heads)
        if self.dim <= 0:
            raise ValueError(f"dim should be positive, got {dim}.")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads should be positive, got {num_heads}.")
        if self.dim % self.num_heads != 0:
            raise ValueError(
                f"dim={self.dim} should be divisible by num_heads={self.num_heads}."
            )

        self.q1 = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.q2 = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.kv1 = nn.Linear(self.dim, self.dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(self.dim, self.dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x1, x2, H=None, W=None):
        B, N, C = _check_token_pair(
            x1,
            x2,
            self.dim,
            "EfficientTokenCrossAttention",
        )
        head_dim = self.dim // self.num_heads

        q1 = self.q1(x1).reshape(B, N, self.num_heads, head_dim).permute(
            0, 2, 1, 3
        )
        q2 = self.q2(x2).reshape(B, N, self.num_heads, head_dim).permute(
            0, 2, 1, 3
        )

        k1, v1 = self.kv1(x1).reshape(
            B, N, 2, self.num_heads, head_dim
        ).permute(2, 0, 3, 1, 4)
        k2, v2 = self.kv2(x2).reshape(
            B, N, 2, self.num_heads, head_dim
        ).permute(2, 0, 3, 1, 4)

        k1 = k1.softmax(dim=-2)
        q1 = q1.softmax(dim=-1)
        ctx1 = k1.transpose(-2, -1) @ v1

        k2 = k2.softmax(dim=-2)
        q2 = q2.softmax(dim=-1)
        ctx2 = k2.transpose(-2, -1) @ v2

        ctx1 = self.attn_drop(ctx1)
        ctx2 = self.attn_drop(ctx2)

        out1 = (q1 @ ctx2).transpose(1, 2).reshape(B, N, C)
        out2 = (q2 @ ctx1).transpose(1, 2).reshape(B, N, C)
        return out1, out2


def build_cross_attention(
    attention_type,
    dim,
    num_heads=1,
    qkv_bias=False,
    attn_drop=0.0,
):
    attention_type = _normalize_attention_type(attention_type)
    if attention_type == "efficient":
        return EfficientTokenCrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
        )
    raise RuntimeError(f"Unsupported attention_type='{attention_type}'.")


class CrossPath(nn.Module):
    """
    Bidirectional cross-modal token interaction path.

    attention_type:
        - "efficient": original efficient attention, O(N * d^2)
    """

    def __init__(
        self,
        dim,
        reduction=4,
        num_heads=1,
        norm_layer=nn.LayerNorm,
        use_cross=True,
        gamma_init=-4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        attention_type="efficient",
        qkv_bias=False,
    ):
        super(CrossPath, self).__init__()
        self.dim = int(dim)
        if reduction <= 0:
            raise ValueError(f"reduction should be positive, got {reduction}.")
        if self.dim <= 0:
            raise ValueError(f"dim should be positive, got {dim}.")

        self.use_cross = use_cross
        self.attention_type = _normalize_attention_type(attention_type)
        if not use_cross:
            return

        if num_heads is None:
            num_heads = 1
        num_heads = int(num_heads)
        if num_heads <= 0:
            raise ValueError(f"num_heads should be positive, got {num_heads}.")
        hidden_dim = max(self.dim // reduction, 1)
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} should be divisible by "
                f"num_heads={num_heads}. Please adjust reduction or num_heads."
            )

        self.hidden_dim = hidden_dim
        self.channel_proj1 = nn.Linear(self.dim, hidden_dim * 2)
        self.channel_proj2 = nn.Linear(self.dim, hidden_dim * 2)
        self.act1 = nn.GELU()
        self.act2 = nn.GELU()

        self.cross_attn = build_cross_attention(
            attention_type=self.attention_type,
            dim=hidden_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
        )
        self.end_proj1 = nn.Linear(hidden_dim * 2, self.dim)
        self.end_proj2 = nn.Linear(hidden_dim * 2, self.dim)

        self.norm1 = norm_layer(self.dim)
        self.norm2 = norm_layer(self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.gamma_logits = nn.Parameter(torch.full((2,), gamma_init))

    def forward(self, x1, x2, H=None, W=None):
        _check_token_pair(x1, x2, self.dim, "CrossPath")
        if not self.use_cross:
            return x1, x2

        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)

        v1, v2 = self.cross_attn(u1, u2, H=H, W=W)

        update1 = self.end_proj1(torch.cat((y1, v1), dim=-1))
        update2 = self.end_proj2(torch.cat((y2, v2), dim=-1))
        update1 = self.proj_drop(self.norm1(update1))
        update2 = self.proj_drop(self.norm2(update2))

        gamma_21, gamma_12 = torch.sigmoid(self.gamma_logits)
        out_x1 = x1 + gamma_21 * update1
        out_x2 = x2 + gamma_12 * update2
        return out_x1, out_x2


class FusionChannelEmbed(nn.Module):
    """
    Channel compression and local spatial mixing after RGB/Event concatenation.

    Input:
        x: [B, N, 2C]
    Output:
        out: [B, C, H, W]
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        reduction=4,
        norm_layer=nn.BatchNorm2d,
    ):
        super(FusionChannelEmbed, self).__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        if reduction <= 0:
            raise ValueError(f"reduction should be positive, got {reduction}.")
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError(
                f"in_channels/out_channels should be positive, "
                f"got {in_channels}/{out_channels}."
            )

        hidden_dim = max(self.out_channels // reduction, 1)

        self.residual = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            bias=False,
        )

        self.channel_embed = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden_dim, kernel_size=1, bias=True),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=hidden_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, self.out_channels, kernel_size=1, bias=True),
        )
        self.norm = norm_layer(self.out_channels)

    def forward(self, x, H, W):
        if x.ndim != 3:
            raise ValueError(
                f"FusionChannelEmbed expects [B, N, C], got {tuple(x.shape)}."
            )
        B, N, C = x.shape
        H = int(H)
        W = int(W)
        if N != H * W:
            raise ValueError(f"Expected N=H*W, got N={N}, H={H}, W={W}.")
        if C != self.in_channels:
            raise ValueError(
                f"FusionChannelEmbed channel mismatch: got C={C}, "
                f"expected {self.in_channels}."
            )

        x = x.transpose(1, 2).reshape(B, C, H, W)

        residual = self.residual(x)
        fused = self.channel_embed(x)
        out = self.norm(residual + fused)
        return out


class CAFFM(nn.Module):
    """
    Attention-based RGB/Event feature fusion module.

    attention_type:
        - "efficient": original efficient cross-attention
    """

    def __init__(
        self,
        dim,
        reduction=4,
        num_heads=1,
        norm_layer=nn.BatchNorm2d,
        use_cross=True,
        gamma_init=-3.0,
        attn_drop=0.0,
        proj_drop=0.0,
        fusion_type="concat",
        fusion_gamma_init=None,
        attention_type="efficient",
        qkv_bias=False,
    ):
        super(CAFFM, self).__init__()
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError(f"dim should be positive, got {dim}.")

        fusion_type = str(fusion_type).strip().lower()
        if fusion_type not in {"concat", "rgb_residual"}:
            raise ValueError(
                f"Unknown fusion_type='{fusion_type}'. "
                "Expected 'concat' or 'rgb_residual'."
            )
        self.fusion_type = fusion_type
        self.attention_type = _normalize_attention_type(attention_type)

        self.cross = CrossPath(
            dim=self.dim,
            reduction=reduction,
            num_heads=num_heads,
            norm_layer=nn.LayerNorm,
            use_cross=use_cross,
            gamma_init=gamma_init,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attention_type=self.attention_type,
            qkv_bias=qkv_bias,
        )
        self.channel_emb = FusionChannelEmbed(
            in_channels=self.dim * 2,
            out_channels=self.dim,
            reduction=reduction,
            norm_layer=norm_layer,
        )

        if self.fusion_type == "rgb_residual":
            if fusion_gamma_init is None:
                fusion_gamma_init = gamma_init
            self.fusion_gamma_logit = nn.Parameter(
                torch.full((1, self.dim, 1, 1), float(fusion_gamma_init))
            )
        else:
            self.register_parameter("fusion_gamma_logit", None)

        self.apply(self._init_weights)
        self.debug_enabled = False
        self.debug_to_cpu = False
        self.debug_cache = {}

    def set_debug(self, enabled: bool = True, to_cpu: bool = False):
        self.debug_enabled = bool(enabled)
        self.debug_to_cpu = bool(to_cpu)
        self.debug_cache = {}
        return self

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        """
        Args:
            x1: RGB feature, [B, C, H, W]
            x2: Event feature, [B, C, H, W]

        Returns:
            Fused feature, [B, C, H, W]
        """
        _check_feature_pair(x1, x2, self.dim, "CAFFM")

        B, C, H, W = x1.shape

        x1_tokens = x1.flatten(2).transpose(1, 2)
        x2_tokens = x2.flatten(2).transpose(1, 2)
        x1_tokens_before = x1_tokens
        x2_tokens_before = x2_tokens

        x1_tokens, x2_tokens = self.cross(x1_tokens, x2_tokens, H=H, W=W)

        merge = torch.cat((x1_tokens, x2_tokens), dim=-1)
        merge = self.channel_emb(merge, H, W)

        if self.fusion_type == "concat":
            out = merge
        elif self.fusion_type == "rgb_residual":
            gamma = torch.sigmoid(self.fusion_gamma_logit)
            out = x1 + gamma * merge
        else:
            raise RuntimeError(f"Unsupported fusion_type='{self.fusion_type}'.")

        if self.debug_enabled:
            self._save_debug_cache(
                x1,
                x2,
                x1_tokens_before,
                x2_tokens_before,
                x1_tokens,
                x2_tokens,
                merge,
                out,
                H,
                W,
            )
        return out

    def _save_debug_cache(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x1_tokens_before: torch.Tensor,
        x2_tokens_before: torch.Tensor,
        x1_tokens_after: torch.Tensor,
        x2_tokens_after: torch.Tensor,
        merge: torch.Tensor,
        out: torch.Tensor,
        height: int,
        width: int,
    ) -> None:
        cache = {
            "rgb_before": _debug_feature_map(x1, self.debug_to_cpu),
            "event_before": _debug_feature_map(x2, self.debug_to_cpu),
            "rgb_before_cross": _debug_token_map(
                x1_tokens_before,
                height,
                width,
                self.debug_to_cpu,
            ),
            "event_before_cross": _debug_token_map(
                x2_tokens_before,
                height,
                width,
                self.debug_to_cpu,
            ),
            "rgb_after_cross": _debug_token_map(
                x1_tokens_after,
                height,
                width,
                self.debug_to_cpu,
            ),
            "event_after_cross": _debug_token_map(
                x2_tokens_after,
                height,
                width,
                self.debug_to_cpu,
            ),
            "fused_update": _debug_feature_map(merge, self.debug_to_cpu),
            "output": _debug_feature_map(out, self.debug_to_cpu),
            "fusion_type": self.fusion_type,
            "attention_type": self.attention_type,
            "use_cross": bool(self.cross.use_cross),
        }
        if self.fusion_gamma_logit is not None:
            cache["fusion_gamma"] = _debug_detach(
                torch.sigmoid(self.fusion_gamma_logit),
                self.debug_to_cpu,
            )
        if getattr(self.cross, "use_cross", False):
            cache["cross_gamma"] = _debug_detach(
                torch.sigmoid(self.cross.gamma_logits),
                self.debug_to_cpu,
            )
        self.debug_cache = cache


class AsymmetricCAFFM(nn.Module):
    """
    Channel-mismatched CAFFM wrapper.

    Event features are first projected into the RGB channel space, then CAFFM
    performs the selected cross-attention and fusion.  The output channel count
    follows the RGB stream, matching the decoder input contract used by CAREFNet.
    """

    def __init__(
        self,
        dim_rgb,
        dim_event,
        reduction=4,
        num_heads=1,
        norm_layer=nn.BatchNorm2d,
        use_cross=True,
        gamma_init=-3.0,
        attn_drop=0.0,
        proj_drop=0.0,
        fusion_type="concat",
        fusion_gamma_init=None,
        attention_type="efficient",
        qkv_bias=False,
    ):
        super().__init__()
        self.dim_rgb = int(dim_rgb)
        self.dim_event = int(dim_event)
        if self.dim_rgb <= 0 or self.dim_event <= 0:
            raise ValueError(
                f"dim_rgb/dim_event should be positive, got "
                f"{dim_rgb}/{dim_event}."
            )

        self.event_to_rgb = nn.Sequential(
            nn.Conv2d(self.dim_event, self.dim_rgb, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.dim_rgb),
        )
        
        self.fusion = CAFFM(
            dim=self.dim_rgb,
            reduction=reduction,
            num_heads=num_heads,
            norm_layer=norm_layer,
            use_cross=use_cross,
            gamma_init=gamma_init,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            fusion_type=fusion_type,
            fusion_gamma_init=fusion_gamma_init,
            attention_type=attention_type,
            qkv_bias=qkv_bias,
        )
        self._init_projection(self.event_to_rgb)
        self.debug_enabled = False
        self.debug_to_cpu = False
        self.debug_cache = {}

    @staticmethod
    def _init_projection(module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def set_debug(self, enabled: bool = True, to_cpu: bool = False):
        self.debug_enabled = bool(enabled)
        self.debug_to_cpu = bool(to_cpu)
        self.fusion.set_debug(enabled, to_cpu=to_cpu)
        self.debug_cache = {}
        return self

    def forward(self, x1, x2):
        if x1.ndim != 4 or x2.ndim != 4:
            raise ValueError(
                f"AsymmetricCAFFM expects NCHW inputs, got "
                f"x1={tuple(x1.shape)}, x2={tuple(x2.shape)}."
            )
        if x1.shape[0] != x2.shape[0] or x1.shape[2:] != x2.shape[2:]:
            raise ValueError(
                "AsymmetricCAFFM expects matched batch/spatial shapes, "
                f"got x1={tuple(x1.shape)}, x2={tuple(x2.shape)}."
            )
        if x1.shape[1] != self.dim_rgb or x2.shape[1] != self.dim_event:
            raise ValueError(
                "AsymmetricCAFFM input channels do not match config, "
                f"got rgb={x1.shape[1]}, event={x2.shape[1]}, "
                f"expected rgb={self.dim_rgb}, event={self.dim_event}."
            )

        x2_projected = self.event_to_rgb(x2)
        out = self.fusion(x1, x2_projected)
        if self.debug_enabled:
            self.debug_cache = dict(getattr(self.fusion, "debug_cache", {}))
            self.debug_cache.update({
                "event_original": _debug_feature_map(x2, self.debug_to_cpu),
                "event_projected": _debug_feature_map(
                    x2_projected,
                    self.debug_to_cpu,
                ),
                "asymmetric": True,
            })
        return out
