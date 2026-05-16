import os
import torch
import matplotlib.pyplot as plt


def save_feature_map(feature, save_path):

    feature = feature.detach().cpu()

    # mean toàn channel
    fmap = torch.mean(
        feature[0],
        dim=0
    ).numpy()

    # normalize
    fmap = (fmap - fmap.min()) / (
        fmap.max() - fmap.min() + 1e-6
    )

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    plt.figure(figsize=(6, 6))

    plt.imshow(fmap, cmap='viridis')

    plt.axis('off')

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()