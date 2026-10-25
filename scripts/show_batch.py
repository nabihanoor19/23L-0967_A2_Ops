import numpy as np
import matplotlib.pyplot as plt
from src.data import make_dataset

ds = make_dataset("data/splits/train.csv", training=True)
images, labels = next(iter(ds))
print("batch shape:", images.shape, "| dtype:", images.dtype,
      "| min/max:", float(images.numpy().min()), float(images.numpy().max()))

plt.figure(figsize=(12, 9))
for i in range(12):
    ax = plt.subplot(3, 4, i + 1)
    plt.imshow(np.clip(images[i].numpy(), 0, 255).astype("uint8"))
    ax.set_title("AI" if labels[i] == 1 else "Real")
    ax.axis("off")
plt.suptitle("One augmented training batch (12 of 32 images)")
plt.tight_layout()
plt.savefig("reports/figures/augmented_batch.png", dpi=120)