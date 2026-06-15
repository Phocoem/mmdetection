import torch
import matplotlib.pyplot as plt
import numpy as np

data = torch.load("feature_maps.pt")

for name in data:

    feat = data[name][0]

    feat = feat.mean(0)

    feat = feat.numpy()

    fft = np.fft.fft2(feat)

    fft = np.fft.fftshift(fft)

    magnitude = np.log(
        np.abs(fft)+1
    )

    plt.imshow(magnitude)
    plt.axis("off")

    plt.savefig(
        f"{name}_fft.png"
    )