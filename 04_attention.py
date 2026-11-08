import math

import torch
from torch import nn


class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, tokens):
        # tokens: [batch, tokens, embedding dimension]
        queries = self.query(tokens)
        keys = self.key(tokens)
        values = self.value(tokens)

        # compare every query with every key
        scores = queries @ keys.transpose(-2, -1)
        scores = scores / math.sqrt(queries.shape[-1])

        # normalize across the keys available to each query
        weights = torch.softmax(scores, dim=-1)

        # gather a weighted combination of value vectors
        output = weights @ values

        return output, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

        self.output_projection = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )

    def split_heads(self, vectors):
        batch_size, num_tokens, _ = vectors.shape

        # divide the projected coordinates into separate heads
        vectors = vectors.reshape(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )

        # place the head axis before the token axis
        return vectors.transpose(1, 2)

    def forward(self, tokens):
        batch_size, num_tokens, embed_dim = tokens.shape

        queries = self.split_heads(self.query(tokens))
        keys = self.split_heads(self.key(tokens))
        values = self.split_heads(self.value(tokens))

        # calculate attention independently for each head
        scores = queries @ keys.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        weights = torch.softmax(scores, dim=-1)
        head_outputs = weights @ values

        # concatenate the heads within each token
        combined = head_outputs.transpose(1, 2).reshape(
            batch_size,
            num_tokens,
            embed_dim,
        )

        output = self.output_projection(combined)

        return output, weights


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_printoptions(precision=3, sci_mode=False)

    # use four small synthetic tokens to inspect the calculation
    tokens = torch.randn(1, 4, 8)

    single_head = SingleHeadAttention(embed_dim=8)
    single_output, single_weights = single_head(tokens)

    print("Single-head output shape:", single_output.shape)
    print("Single-head attention weights:")
    print(single_weights[0].detach())

    multi_head = MultiHeadAttention(embed_dim=8, num_heads=2)
    multi_output, multi_weights = multi_head(tokens)

    print("\nMulti-head output shape:", multi_output.shape)
    print("Multi-head weights shape:", multi_weights.shape)

    for head_index in range(2):
        print(f"\nHead {head_index} attention weights:")
        print(multi_weights[0, head_index].detach())

    # each query distributes a total weight of one across keys
    assert torch.allclose(
        single_weights.sum(dim=-1),
        torch.ones_like(single_weights.sum(dim=-1)),
    )

    assert torch.allclose(
        multi_weights.sum(dim=-1),
        torch.ones_like(multi_weights.sum(dim=-1)),
    )

    assert single_output.shape == tokens.shape
    assert multi_output.shape == tokens.shape

    print("\nAttention checks passed.")