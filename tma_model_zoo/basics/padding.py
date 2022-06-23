import torch.nn.functional as functional


def same_padding(x, y):
    height, width = y.shape[2], y.shape[3]

    # input is CHW
    diff_y = height - x.shape[2]
    diff_x = width - x.shape[3]

    if diff_y != 0 or diff_x != 0:
        padding_size = [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        x = functional.pad(x, padding_size)

    return x
