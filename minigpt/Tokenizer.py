# Tokenizer.py
# A simple character-level tokenizer.

import torch

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Build the vocabulary: every unique character, in a fixed order
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Build lookup tables in both directions
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    """Convert a string into a list of integer token IDs."""
    return [stoi[c] for c in s]

def decode(ids):
    """Convert a list of integer token IDs back into a string."""
    return "".join(itos[i] for i in ids)

# Encode the entire dataset as a tensor (reusable by other files, like train.py later)
data = torch.tensor(encode(text), dtype=torch.long)


# ---- Everything below only runs when you do `python Tokenizer.py` directly ----
if __name__ == "__main__":
    print("Length of dataset in characters:", len(text))
    print(text[:200])

    print("\nVocabulary size:", vocab_size)
    print("All characters:", "".join(chars))

    sample = "Hi there!"
    ids = encode(sample)
    back = decode(ids)

    print("\nOriginal:", sample)
    print("Encoded: ", ids)
    print("Decoded: ", back)
    assert back == sample, "Round-trip failed!"
    print("Round-trip successful!")

    print("\nFull dataset encoded as tensor:")
    print("Shape:", data.shape)
    print("Dtype:", data.dtype)
    print("First 20 token IDs:", data[:20].tolist())