import csv
import random
from collections import Counter, defaultdict

import torch
from PIL import Image

from prototype import load_model, transform, CLASSES

TEST_CSV = r"D:\download\review2\outputs\test.csv"
PER_CLASS = 20
SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(device)

# read test.csv, keep only A-Z (the model has 26 classes)
by_class = defaultdict(list)
with open(TEST_CSV, newline="") as f:
    for row in csv.DictReader(f):
        if row["label_name"] in CLASSES:
            by_class[row["label_name"]].append(row["path"])

rng = random.Random(SEED)
total = 0
correct = 0
per_class_correct = {}
wrong_pairs = Counter()

for name in CLASSES:
    paths = by_class[name]
    sample = rng.sample(paths, min(PER_CLASS, len(paths)))
    c = 0
    for p in sample:
        img = Image.open(p).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = CLASSES[model(x).argmax(dim=1).item()]
        total += 1
        if pred == name:
            c += 1
            correct += 1
        else:
            wrong_pairs[(name, pred)] += 1
    per_class_correct[name] = (c, len(sample))

print("Images checked :", total, "(A-Z only,", PER_CLASS, "per letter, from test.csv)")
print(f"Correct        : {correct}/{total}  = {100 * correct / total:.1f}%")
print()
print("Per-letter correct:")
print("  " + "  ".join(f"{k}:{v[0]}/{v[1]}" for k, v in per_class_correct.items()))
print()
print("Most common mistakes (true -> predicted):")
for (t, p), n in wrong_pairs.most_common(8):
    print(f"  {t} -> {p} : {n} times")