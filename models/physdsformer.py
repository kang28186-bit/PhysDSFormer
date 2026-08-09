import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import _calculate_fan_in_and_fan_out, trunc_normal_


class RLN(nn.Module):
    r"""Revised LayerNorm"""

    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)

        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias


class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        return self.mlp(x)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        B_, N, _ = qkv.shape

        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=1, padding=padding, groups=in_channels, padding_mode='reflect')
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(x)

class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.network_depth = network_depth
        self.QK = nn.Conv2d(dim, dim * 2, 1)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:  # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
                      mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X):
        B, C, H, W = X.shape

        QK = self.QK(X)
        QKV = torch.cat([QK, X], dim=1)
        shifted_QKV = self.check_size(QKV, self.shift_size > 0)
        Ht, Wt = shifted_QKV.shape[2:]
        shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
        qkv = window_partition(shifted_QKV, self.window_size)
        attn_windows = self.attn(qkv)
        shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)
        out = shifted_out[
            :, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W), :
        ]
        return out.permute(0, 3, 1, 2)

class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm

        if conv_type != 'DSC':
            raise ValueError("PhysDSFormer blocks require conv_type='DSC'")
        self.local_enhancement = DepthwiseSeparableConv(
            dim, dim, kernel_size=3, padding=1
        )

        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = (
            Attention(
                network_depth,
                dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
            )
            if use_attn
            else None
        )

        self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        x = self.local_enhancement(x)

        if self.use_attn:
            identity = x
            x, rescale, rebias = self.norm1(x)
            x = self.attn(x)
            x = identity + x * rescale + rebias

        identity = x
        if self.use_attn and self.mlp_norm:
            x, rescale, rebias = self.norm2(x)
        x = self.mlp(x)
        if self.use_attn and self.mlp_norm:
            x = x * rescale + rebias
        x = identity + x
        return x


class BasicLayer(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, window_size=8,
                 attn_ratio=0., attn_loc='last', conv_type=None):

        super().__init__()
        self.dim = dim
        self.depth = depth

        attn_depth = attn_ratio * depth

        if attn_loc == 'last':
            use_attns = [i >= depth - attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        elif attn_loc == 'middle':
            use_attns = [i >= (depth - attn_depth) // 2 and i < (depth + attn_depth) // 2 for i in range(depth)]

        # build blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(network_depth=network_depth,
                             dim=dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             window_size=window_size,
                             shift_size=0 if (i % 2 == 0) else window_size // 2,
                             use_attn=use_attns[i], conv_type=conv_type)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class OPReconstructionModule(nn.Module):
    """Estimate channel-wise gain (O) and bias (P) from an underwater image."""

    def __init__(self, in_channels=3, hidden_channels=7):
        super().__init__()
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, padding_mode='reflect'),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 5, padding=2, padding_mode='reflect'),
            nn.Conv2d(hidden_channels, hidden_channels, 5, padding=2, padding_mode='reflect'),
            nn.ReLU(inplace=True),
        )
        self.shortcut = nn.Conv2d(
            in_channels, hidden_channels, 5, padding=2, padding_mode='reflect'
        )
        self.parameter_head = nn.Conv2d(
            hidden_channels, in_channels * 2, 3, padding=1, padding_mode='reflect'
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, image):
        parameters = self.parameter_head(self.main_branch(image) + self.shortcut(image))
        gain, bias = torch.chunk(parameters, 2, dim=1)
        return gain, bias


class PhysicsAwarePatchEmbedding(nn.Module):
    """Apply the simplified physical restoration and project it with a 5x5 convolution."""

    def __init__(self, in_channels=3, embed_dim=24, patch_size=1):
        super().__init__()
        self.reconstruction = OPReconstructionModule(in_channels=in_channels)
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=5,
            stride=patch_size,
            padding=2,
            padding_mode='reflect',
        )
        trunc_normal_(self.projection.weight, std=0.02)
        nn.init.constant_(self.projection.bias, 0)

    def forward(self, image):
        gain, bias = self.reconstruction(image)
        restored = gain * image + bias
        return self.projection(restored), restored

class PhysDSFormer(nn.Module):
    def __init__(self, in_chans=3, out_chans=4, window_size=8,
                 embed_dims=(24, 48, 96, 48),
                 mlp_ratios=(2., 4., 4., 2.),
                 depths=(16, 16, 16, 8),
                 num_heads=(2, 4, 6, 1),
                 attn_ratio=(1 / 4, 1 / 2, 3 / 4, 0),
                 conv_type=('DSC', 'DSC', 'DSC', 'DSC'),
                 norm_layer=(RLN, RLN, RLN, RLN)):
        super().__init__()

        # setting
        self.patch_size = 4
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios

        self.physpe = PhysicsAwarePatchEmbedding(
            in_channels=in_chans, embed_dim=embed_dims[0], patch_size=1
        )

        # backbone
        self.layer1 = BasicLayer(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
                                 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                 norm_layer=norm_layer[0], window_size=window_size,
                                 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
                                 num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                                 norm_layer=norm_layer[1], window_size=window_size,
                                 attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
                                 num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                 norm_layer=norm_layer[2], window_size=window_size,
                                 attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.layer4 = BasicLayer(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
                                 num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                 norm_layer=norm_layer[3], window_size=window_size,
                                 attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[0], embed_dim=embed_dims[3])

        self.fusion2 = SKFusion(embed_dims[0])

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[0], kernel_size=3)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        minimum_size = self.patch_size * self.window_size
        target_h = max(minimum_size, math.ceil(h / self.patch_size) * self.patch_size)
        target_w = max(minimum_size, math.ceil(w / self.patch_size) * self.patch_size)
        pad_h = target_h - h
        pad_w = target_w - w
        padding_mode = 'reflect' if h > pad_h and w > pad_w else 'replicate'
        x = F.pad(x, (0, pad_w, 0, pad_h), padding_mode)
        return x

    def forward_features(self, x):
        x, restored = self.physpe(x)

        x = self.layer1(x)
        skip1 = x

        x = self.patch_merge1(x)
        x = self.layer2(x)
        skip2 = x

        x = self.patch_merge2(x)
        x = self.layer3(x)
        x = self.patch_split1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)
        x = self.patch_split2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.patch_unembed(x)
        return x, restored

    def forward(self, x):
        height, width = x.shape[2:]
        x = self.check_image_size(x)
        parameters, restored = self.forward_features(x)
        gain, bias = torch.split(parameters, (1, 3), dim=1)
        output = gain * restored + bias + restored
        return output[:, :, :height, :width]


def physdsformer_tiny():
    return PhysDSFormer(
        embed_dims=(24, 24, 24, 24),
        mlp_ratios=(2., 2., 2., 2.),
        depths=(2, 2, 2, 2),
        num_heads=(2, 2, 2, 1),
        attn_ratio=(0, 1 / 2, 1, 0),
        conv_type=('DSC', 'DSC', 'DSC', 'DSC'))


def physdsformer_base():
    return PhysDSFormer(
        embed_dims=(24, 48, 96, 48),
        mlp_ratios=(2., 4., 4., 2.),
        depths=(8, 8, 8, 8),
        num_heads=(2, 4, 6, 2),
        attn_ratio=(1 / 4, 1 / 2, 3 / 4, 0),
        conv_type=('DSC', 'DSC', 'DSC', 'DSC'))
