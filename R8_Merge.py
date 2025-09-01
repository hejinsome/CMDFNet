import torch
from R8_Scan import diagonal_scan,R8_Scan


# ---------- R8_Merge ----------
def R8_Merge(y, H, W):
    B, HW, _, C = y.shape

    outs = y.new_empty((B, 8, C, H, W))

    # 0/1: Row scan + reverse row scan (the reverse sequence must be flipped back before reconstruction)
    outs[:, 0] = y[:, :, 0, :].transpose(1, 2).reshape(B, C, H, W)
    outs[:, 1] = torch.flip(y[:, :, 1, :], dims=[1]).transpose(1, 2).reshape(B, C, H, W)

    # 2/3: Column scan + reverse column scan (the reverse sequence must be flipped back before reconstruction)
    seq = y[:, :, 2, :]
    outs[:, 2] = seq.transpose(1, 2).reshape(B, C, W, H).transpose(2, 3)
    seq = torch.flip(y[:, :, 3, :], dims=[1])
    outs[:, 3] = seq.transpose(1, 2).reshape(B, C, W, H).transpose(2, 3)

    # 4/5: lr diagonal + reverse lr (the reverse sequence must be flipped first, then restored using the inverse index mapping)
    idx = diagonal_scan(H, W, direction="lr")
    inv_lr = torch.argsort(torch.tensor(idx, device=y.device))
    seq = y[:, :, 4, :][:, inv_lr, :]
    outs[:, 4] = seq.transpose(1, 2).reshape(B, C, H, W)
    seq = torch.flip(y[:, :, 5, :], dims=[1])[:, inv_lr, :]
    outs[:, 5] = seq.transpose(1, 2).reshape(B, C, H, W)

    # 6/7: rl diagonal + reverse rl (the reverse sequence must be flipped first, then restored using the inverse index mapping)
    idx = diagonal_scan(H, W, direction="rl")
    inv_idx = torch.argsort(torch.tensor(idx))
    seq = y[:, :, 6, :][:, inv_idx, :]
    outs[:, 6] = seq.transpose(1, 2).reshape(B, C, H, W)
    seq = torch.flip(y[:, :, 7, :], dims=[1])[:, inv_idx, :]
    outs[:, 7] = seq.transpose(1, 2).reshape(B, C, H, W)


    #out = torch.stack(outs, dim=0).sum(0) / 8.0
    return outs

'''
if __name__ == '__main__':
    # ---------- Test with 3x3 image ----------
    x = torch.arange(9).reshape(1, 1, 3, 3).float()
    print("Original image:\n", x[0, 0])

    y = R8_Scan(x)

    # Print the 8 directional scanning sequences
    directions = ["row", "row_rev", "col", "col_rev", "lr", "lr_rev", "rl", "rl_rev"]
    print("\n--- 8 directional scanning sequences ---")
    for i, d in enumerate(directions):
        print(f"{d}: {y[0, :, i, 0].tolist()}")

    # Reconstruct the image from each direction
    outs = R8_Merge(y, 3, 3)
    print("\n--- Reconstructed images for each direction ---")
    for i, d in enumerate(directions):
        print(f"{d} reconstruction:\n", outs[0, i, 0].int())

'''
