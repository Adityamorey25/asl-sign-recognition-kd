import os
import sys
import time

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ---------------- Settings ----------------
REPO_DIR = r"D:\LY_Project_SLRSystem--KD_DL"
STUDENT_PATH = os.path.join(REPO_DIR, "model_weights", "student_best_aug.pth")
OUT_DIR = r"D:\download\review2\outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Base repo model is trained on A-Z only (26 classes)
CLASSES = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
NUM_CLASSES = len(CLASSES)

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ---------------- Model (plug-in part) ----------------
# Same structure as the base repo student model, so weights load correctly.
# To plug in a new model later, only this class and load_model() change.
class MobileNetV2Student(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.last_channel, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def load_model(device):
    if not os.path.exists(STUDENT_PATH):
        print("ERROR: weights not found at", STUDENT_PATH)
        sys.exit(1)
    model = MobileNetV2Student(NUM_CLASSES).to(device)
    state = torch.load(STUDENT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------- Stage 1: Preprocessing ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def preprocess(bgr_image, device):
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)   # OpenCV gives BGR
    pil = Image.fromarray(rgb)
    return transform(pil).unsqueeze(0).to(device)       # shape (1, 3, 224, 224)


# ---------------- Stage 2: Recognition ----------------
def predict(model, tensor):
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    conf, idx = torch.max(probs, dim=0)
    top3 = torch.topk(probs, 3)
    top3_list = [(CLASSES[i], float(p)) for p, i in zip(top3.values, top3.indices)]
    return CLASSES[idx.item()], float(conf), top3_list


# ---------------- Mode 1: single image ----------------
def run_image(path, model, device):
    img = cv2.imread(path)
    if img is None:
        print("ERROR: could not read image:", path)
        sys.exit(1)
    print("Input image      :", path)
    print("Original size    : (H, W, C) =", img.shape)
    tensor = preprocess(img, device)
    print("After preprocess :", tuple(tensor.shape))
    label, conf, top3 = predict(model, tensor)
    print(f"Predicted sign   : {label}  (model confidence {conf * 100:.1f}%)")
    print("Top-3            :", ", ".join(f"{c} {p * 100:.1f}%" for c, p in top3))

    shown = cv2.resize(img, (400, 400))
    cv2.putText(shown, f"Pred: {label} ({conf * 100:.0f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    out_path = os.path.join(OUT_DIR, "prototype_image_result.png")
    cv2.imwrite(out_path, shown)
    print("Saved result     :", out_path)
    cv2.imshow("Prototype - image mode (press any key to close)", shown)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------- Mode 2: webcam ----------------
def run_webcam(model, device):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: webcam could not be opened.")
        print("Use image mode instead:  python prototype.py <image_path>")
        sys.exit(1)

    print("Webcam started. Keep your hand inside the green box.")
    print("Keys:  q = quit   s = save snapshot")
    prev_time = time.time()
    fps = 0.0
    snap_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: could not read frame from webcam.")
            break

        h, w = frame.shape[:2]
        box = int(min(h, w) * 0.6)
        x1 = (w - box) // 2
        y1 = (h - box) // 2
        x2, y2 = x1 + box, y1 + box
        roi = frame[y1:y2, x1:x2]

        tensor = preprocess(roi, device)
        label, conf, _ = predict(model, tensor)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Prediction: {label}  ({conf * 100:.0f}%)", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "Flow: Camera > Preprocess > Student model > Prediction",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # preview of the preprocessed input (224x224) in the corner
        if w > 200:
            thumb = cv2.resize(roi, (IMG_SIZE // 2, IMG_SIZE // 2))
            th = thumb.shape[0]
            frame[10:10 + th, w - th - 10:w - 10] = thumb
            cv2.putText(frame, "model input", (w - th - 10, 10 + th + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("ASL Prototype - Review II", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            snap_count += 1
            p = os.path.join(OUT_DIR, f"prototype_snapshot_{snap_count}.png")
            cv2.imwrite(p, frame)
            print("Saved snapshot:", p)

    cap.release()
    cv2.destroyAllWindows()


# ---------------- Main ----------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device :", device)
    model = load_model(device)
    print("Student model loaded:", STUDENT_PATH)
    print("Classes:", NUM_CLASSES, "(A-Z)")
    print()

    if len(sys.argv) > 1:
        run_image(sys.argv[1], model, device)
    else:
        run_webcam(model, device)