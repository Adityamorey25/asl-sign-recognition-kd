import os
from collections import OrderedDict
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_DIR = r"D:\download\z total project\archive\asl_alphabet_train\asl_alphabet_train"
OUT_DIR = r"D:\download\review2\outputs"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

classes = sorted(
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
)

counts = OrderedDict()
first_image = {}
for c in classes:
    files = [f for f in os.listdir(os.path.join(DATASET_DIR, c))
             if f.lower().endswith(IMG_EXT)]
    counts[c] = len(files)
    if files:
        first_image[c] = os.path.join(DATASET_DIR, c, sorted(files)[0])

print("Dataset folder :", DATASET_DIR)
print("Total classes  :", len(classes))
print("Classes        :", classes)
print("Total images   :", sum(counts.values()))
print()
print("Class-wise image count:")
for c, n in counts.items():
    print(f"  {c:>8} : {n}")

# Verify loading
failed = 0
for c, p in first_image.items():
    try:
        with Image.open(p) as im:
            im.load()
    except Exception as e:
        failed += 1
        print("LOAD FAILED:", p, e)
print()
print("Load check: sample image opened successfully for",
      len(first_image) - failed, "of", len(classes), "classes")
with Image.open(next(iter(first_image.values()))) as im:
    print("Sample image size (W x H) and mode:", im.size, im.mode)

# Class distribution chart
plt.figure(figsize=(12, 5))
plt.bar(list(counts.keys()), list(counts.values()))
plt.xticks(rotation=90)
plt.ylabel("Number of images")
plt.title("Class-wise distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=150)
plt.close()

# Sample image grid (one per class)
cols = 8
rows = (len(first_image) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 2.2 * rows))
for ax in axes.flat:
    ax.axis("off")
for ax, (c, p) in zip(axes.flat, first_image.items()):
    with Image.open(p) as im:
        ax.imshow(im.convert("RGB"))
    ax.set_title(c, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "sample_images.png"), dpi=120)
plt.close()

print()
print("Saved charts in:", OUT_DIR)