import torch
from torch import nn

from attention import MultiHeadAttention


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = embed_dim * mlp_ratio

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, tokens):
        # exchange information between normalized tokens
        attention_output, _ = self.attention(self.norm1(tokens))

        # add the attention update to the incoming tokens
        tokens = tokens + attention_output

        # transform each token independently and add its update
        tokens = tokens + self.mlp(self.norm2(tokens))

        return tokens


if __name__ == "__main__":
    torch.manual_seed(42)

    tokens = torch.randn(2, 64, 64)

    block = TransformerEncoderBlock(
        embed_dim=64,
        num_heads=4,
    )

    output = block(tokens)

    print("Input shape:", tokens.shape)
    print("Output shape:", output.shape)

    assert output.shape == tokens.shape

    # verify that gradients reach every trainable parameter
    loss = output.square().mean()
    loss.backward()

    assert all(
        parameter.grad is not None
        for parameter in block.parameters()
        if parameter.requires_grad
    )

    print("Encoder checks passed.")