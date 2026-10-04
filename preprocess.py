import os
import csv
import json
import random

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_DIR = r"D:\download\z total project\archive\asl_alphabet_train\asl_alphabet_train"
OUT_DIR = r"D:\download\review2\outputs"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1   # remaining 0.1 = test


# ---------- 1. Label encoding ----------
classes = sorted(
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
)
label_map = {name: idx for idx, name in enumerate(classes)}
with open(os.path.join(OUT_DIR, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=2)
print("Label encoding (class -> id):")
print(label_map)
print()


# ---------- 2. Train / Val / Test split (stratified, per class) ----------
rng = random.Random(SEED)
splits = {"train": [], "val": [], "test": []}
for name in classes:
    folder = os.path.join(DATASET_DIR, name)
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))
    rng.shuffle(files)
    n = len(files)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    parts = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }
    for split_name, flist in parts.items():
        for fn in flist:
            splits[split_name].append((os.path.join(folder, fn), name, label_map[name]))

for split_name, rows in splits.items():
    csv_path = os.path.join(OUT_DIR, f"{split_name}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label_name", "label_id"])
        w.writerows(rows)
    print(f"{split_name:>5} split : {len(rows)} images -> {csv_path}")
print()


# ---------- 3. Preprocessing transform (same as base repo) ----------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),                       # 0-255 -> 0.0-1.0, HWC -> CHW
    transforms.Normalize(mean=MEAN, std=STD),
])


class ASLDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, _, label_id = self.rows[i]
        img = Image.open(path).convert("RGB")    # PIL gives RGB already
        return self.transform(img), label_id


# ---------- 4. Before / after demo on real images ----------
def denormalize(t):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


demo_classes = ["A", "M", "Z"]
fig, axes = plt.subplots(len(demo_classes), 2, figsize=(7, 3.2 * len(demo_classes)))
print("Before / After preprocessing:")
for r, cname in enumerate(demo_classes):
    row = next(x for x in splits["train"] if x[1] == cname)
    path, name, lid = row
    with Image.open(path) as im:
        orig = im.convert("RGB")
        orig_size = orig.size
        orig_copy = orig.copy()
    tensor = transform(orig_copy)
    print(f"  Class '{name}' -> encoded label {lid}")
    print(f"    Original size (W x H) : {orig_size}")
    print(f"    After  shape  (C,H,W) : {tuple(tensor.shape)}")
    print(f"    After  value range    : min={tensor.min().item():.3f}, max={tensor.max().item():.3f}")
    axes[r][0].imshow(orig_copy)
    axes[r][0].set_title(f"Original {orig_size[0]}x{orig_size[1]} | '{name}' (label {lid})", fontsize=9)
    axes[r][1].imshow(denormalize(tensor))
    axes[r][1].set_title(f"Preprocessed {IMG_SIZE}x{IMG_SIZE} (normalized)", fontsize=9)
    axes[r][0].axis("off")
    axes[r][1].axis("off")
plt.tight_layout()
demo_path = os.path.join(OUT_DIR, "preprocessing_demo.png")
plt.savefig(demo_path, dpi=130)
plt.close()
print("Saved:", demo_path)
print()


# ---------- 5. Check: data is ready for the next stage ----------
loader = DataLoader(ASLDataset(splits["train"], transform),
                    batch_size=32, shuffle=True, num_workers=0)
images, labels = next(iter(loader))
print("DataLoader check (one training batch):")
print("  images tensor shape :", tuple(images.shape))
print("  labels tensor shape :", tuple(labels.shape))
print("  first 10 labels     :", labels[:10].tolist())
print()
print("Preprocessing pipeline OK.")