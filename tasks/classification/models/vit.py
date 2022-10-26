import torch
import numpy as np
import torchsummary
import torchvision
from einops.layers.torch import Rearrange, Reduce
from einops import repeat, rearrange


class ViT(torch.nn.Sequential):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224, depth=12, **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            Transformer(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, 10)
        )


class PatchEmbedding(torch.nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=112):
        super().__init__()
        self.sequence = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, emb_size, patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e')
        )

        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos = torch.nn.Parameter(torch.randn((img_size // patch_size)**2+1, emb_size))

    def forward(self, x):
        b = x.shape[0]
        x = self.sequence(x)
        cls_token = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_token, x], dim=1)
        x += self.pos
        return x


class Transformer(torch.nn.Sequential):
    def __init__(self, depth=12, **kwargs):
        super().__init__(
            *[EncoderBlock(**kwargs) for _ in range(depth)]
        )


class ClassificationHead(torch.nn.Sequential):
    def __init__(self, emb_size=768, n_classes=10):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            torch.nn.LayerNorm(emb_size),
            torch.nn.Linear(emb_size, n_classes))


class EncoderBlock(torch.nn.Module):
    def __init__(self, emb_size=768, forward_expansion=4, **kwargs):
        super().__init__(
            Residual(torch.nn.Sequential(
                torch.nn.LayerNorm(emb_size),
                MSA(emb_size, **kwargs),
            )),
            Residual(torch.nn.Sequential(
                torch.nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion),
            ))
        )


class Residual(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(torch.nn.Sequential):
    def __init__(self, emb_size, expansion=4):
        super().__init__(
            torch.nn.Linear(emb_size, expansion * emb_size),
            torch.nn.GELU(),
            torch.nn.Linear(expansion * emb_size, emb_size)
        )


class MSA(torch.nn.Module):
    def __init__(self, emb_size=768, num_heads=8):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.q = torch.nn.Linear(emb_size, emb_size)
        self.k = torch.nn.Linear(emb_size, emb_size)
        self.v = torch.nn.Linear(emb_size, emb_size)
        self.projection = torch.nn.Linear(emb_size, emb_size)


    def forward(self, x):
        q = rearrange(self.q(x), 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(self.k(x), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.v(x), 'b n (h d) -> b h n d', h=self.num_heads)
        score = torch.einsum('bhqd, bhkd -> bhqk', q, k)

        scaling = self.emb_size ** .5
        attention = torch.softmax(score, dim=-1) / scaling
        out = torch.einsum('bhal, bhlv -> bhav', attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        return out


