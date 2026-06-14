"""Next-token prediction dataset: random contiguous BPE token windows."""

from __future__ import annotations

import torch


class GPTDataset:
    """
    Random slices from a long token ID stream.

    Example tokens [10, 11, 12, 13, 14, 15]:
      input  x = [10, 11, 12, 13, 14]
      target y = [11, 12, 13, 14, 15]
    """

    def __init__(self, token_ids: list[int], vocab_size: int) -> None:
        if len(token_ids) < 2:
            raise ValueError("Need at least 2 tokens for next-token prediction")
        self.vocab_size = vocab_size
        self.data = torch.tensor(token_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(
        self, batch_size: int, block_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - block_size - 1
        if max_start < 0:
            raise ValueError(
                f"Data length {len(self.data)} too short for block_size {block_size}"
            )
        starts = torch.randint(0, max_start + 1, (batch_size,))
        x = torch.stack([self.data[s : s + block_size] for s in starts])
        y = torch.stack([self.data[s + 1 : s + block_size + 1] for s in starts])
        return x, y


def split_token_ids(
    ids: list[int], val_ratio: float = 0.1
) -> tuple[list[int], list[int]]:
    split_at = int(len(ids) * (1.0 - val_ratio))
    return ids[:split_at], ids[split_at:]
