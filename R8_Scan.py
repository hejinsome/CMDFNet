import torch
def diagonal_scan(H, W, direction="lr"):
    '''
    direction:
    'lr'(↘ top-left → bottom-right),
    'rl'(↙ top-right → bottom-left)
    '''
    seq = []
    if direction in ["lr"]:
        for diagon in range(H + W - 1):
            row = 0 if diagon < W else diagon - W + 1
            col = diagon if diagon < W else W - 1
            while row < H and col >= 0:
                idx = row * W + col
                seq.append(idx)
                row += 1
                col -= 1
    elif direction in ["rl"]:
        for diagon in range(H + W - 1):
            row = 0 if diagon < W else diagon - W + 1
            col = W - 1 - diagon if diagon < W else 0
            while row < H and col < W:
                idx = row * W + col
                seq.append(idx)
                row += 1
                col += 1
    return seq
def R8_Scan(x):
    B, C, H, W = x.shape
    y = x.new_empty((B, 8, C, H * W))
    y[:, 0] = x.flatten(2, 3)
    y[:, 1] = torch.flip(y[:, 0], dims=[-1])
    y[:, 2] = x.transpose(2, 3).flatten(2, 3)
    y[:, 3] = torch.flip(y[:, 2], dims=[-1])
    idx = diagonal_scan(H, W, direction="lr")
    y[:, 4] = x.flatten(2, 3)[:, :, idx]
    y[:, 5] = torch.flip(y[:, 4], dims=[-1])
    idx = diagonal_scan(H, W, direction="rl")
    y[:, 6] = x.flatten(2, 3)[:, :, idx]
    y[:, 7] = torch.flip(y[:, 6], dims=[-1])
    y = y.permute(0, 3, 1, 2).contiguous()  # (B,H*W,8,C)
    return y
