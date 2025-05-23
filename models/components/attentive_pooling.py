import torch 
import torch.nn as nn
import torch.nn.functional as F

class AttentivePooling(nn.Module):
    # taken from OG paper: https://github.com/apple/ml-aim/blob/main/aim-v1/aim/v1/torch/layers.py
    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        num_queries: int = 1,
        use_batch_norm: bool = True,
        qkv_bias: bool = False,
        average_pool: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.average_pool = average_pool

        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.cls_token = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        # self.bn = (
        #     nn.BatchNorm1d(dim, affine=False, eps=1e-6)
        #     if use_batch_norm
        #     else nn.Identity()
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 1:, :]  # exclude the ViT CLS token (could also be done in code later if neccessary)
        B, N, C = x.shape
        #x = self.bn(x.transpose(-2, -1)).transpose(-2, -1) #done with fc_norm later
        cls_token = self.cls_token.expand(B, -1, -1)

        q = cls_token.reshape(
            B, self.num_queries, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        x_cls = F.scaled_dot_product_attention(q, k, v)
        x_cls = x_cls.transpose(1, 2).reshape(B, self.num_queries, C)
        x_cls = x_cls.mean(dim=1) if self.average_pool else x_cls
        return x_cls