# embeddings.py
# Turning token IDs into learnable vectors.

import torch
import torch.nn as nn
from tokenizer import vocab_size, encode, stoi

# Embedding dimension - how many numbers represent each token.
# Real GPTs use hundreds/thousands; we start small so it's easy to inspect.
n_embd = 32

# nn.Embedding creates the lookup table: shape (vocab_size, n_embd),
# initialized with random values (these are "weights" - learnable parameters).
token_embedding_table = nn.Embedding(vocab_size, n_embd)


if __name__ == "__main__":
    print("Embedding table shape:", token_embedding_table.weight.shape)
    print("Total weights in table:", token_embedding_table.weight.numel())

    # Take a small sample of token IDs using our real tokenizer
    sample_text = "Hi"
    sample_ids = torch.tensor(encode(sample_text))
    print("\nToken IDs for 'Hi':", sample_ids)

    # Look up their embeddings
    embeddings = token_embedding_table(sample_ids)
    print("Embeddings shape:", embeddings.shape)
    print("Embedding vector for 'H':", embeddings[0])
    print("Embedding vector for 'i':", embeddings[1])

    # Confirm: calling the embedding layer is the same as indexing the table directly
    print("\nRow 47 directly:", token_embedding_table.weight[47])
    print("Row 47 via lookup:", token_embedding_table(torch.tensor([47])))

