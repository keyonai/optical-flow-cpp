# ─────────────────────────────────────────────────────────────
# export_raft.py
# Exports the RAFT-small optical flow model to TorchScript.
# Run this ONCE in Python before building the C++ project.
# Output: raft_small.pt  (loaded by main.cpp via LibTorch)
# ─────────────────────────────────────────────────────────────

import torch
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights


class RAFTInference(torch.nn.Module):
    """
    Thin wrapper around torchvision's RAFT-small model.

    RAFT internally runs 12 recurrent update steps and returns all
    intermediate flow predictions as a list. This wrapper returns
    only the FINAL prediction (index 11) as a plain Tensor —
    much easier to consume from C++.

    Input:   two float tensors [1, 3, H, W] with pixel values in [0, 1]
    Output:  flow tensor [1, 2, H, W]
               channel 0 = horizontal displacement (pixels)
               channel 1 = vertical   displacement (pixels)
    """

    def __init__(self):
        super().__init__()
        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        predictions = self.model(img1, img2)   # list of 12 tensors
        return predictions[11]                  # final estimate [1, 2, H, W]


# ── Load weights ────────────────────────────────────────────
print("Loading RAFT-small pretrained weights...")
model = RAFTInference()
model.eval()

# ── Trace with dummy inputs ──────────────────────────────────
# Dimensions MUST be divisible by 8 — RAFT's feature pyramid requires it.
# 360x480 matches a 4:3 webcam crop and is divisible by 8.
H, W = 360, 480
img1 = torch.zeros(1, 3, H, W)
img2 = torch.zeros(1, 3, H, W)

print(f"Tracing model at {H}x{W} resolution...")
with torch.no_grad():
    traced = torch.jit.trace(model, (img1, img2))

traced.save("raft_small.pt")
print("Saved  →  raft_small.pt")

# ── Verify the saved model loads and runs correctly ──────────
print("Verifying export...")
loaded = torch.jit.load("raft_small.pt")
loaded.eval()
with torch.no_grad():
    out = loaded(img1, img2)
print(f"Output shape: {out.shape}")   # should be [1, 2, 360, 480]
print("Export successful!")
