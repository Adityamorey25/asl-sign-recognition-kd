# ASL Recognition using Knowledge Distillation - Review II (50% Implementation)

Final year project, Group 21.
Review II scope: Dataset, Data Preprocessing, Prototype.

Base / reference repository: github.com/AayushiWani/LY_Project_SLRSystem--KD_DL
(by Aayushi Wani and Harsh, shared with our group for this project).
Its student model weights and preprocessing settings are reused here as a baseline.
Everything in this folder is our own work on top of it.

Pipeline:

    Dataset -> Data Preprocessing -> Prototype -> Basic Sign/Class Prediction

Note: the dataset (87,000 images) and the model weights are NOT included in this
repository because of their size. Paths inside the scripts point to the local machine.


## 1. Dataset

- Folder: D:\download\z total project\archive\asl_alphabet_train\asl_alphabet_train
- Script: dataset_check.py
- Classes: 29 (A-Z, del, nothing, space)
- Total images: 87,000 (exactly 3,000 per class, so the dataset is balanced)
- Image size: 200 x 200, RGB
- Load check: a sample image opened successfully for all 29 classes
- Outputs: outputs\class_distribution.png, outputs\sample_images.png

Run:

    D:\LY_Project_SLRSystem--KD_DL\venv\Scripts\python.exe dataset_check.py


## 2. Data Preprocessing

- Script: preprocess.py
- Steps: read image (RGB) -> resize to 224 x 224 -> convert to tensor (0-1)
  -> normalize with ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]
  (same settings as the base repo, so a model trained on it stays compatible)
- Label encoding: A=0 ... Z=25, del=26, nothing=27, space=28 (outputs\label_map.json)
- Split: stratified 80/10/10 per class, seed 42
  - train 69,600 | val 8,700 | test 8,700 (train.csv, val.csv, test.csv)
- Result check: original (200, 200) -> tensor (3, 224, 224), values from -2.118 to 2.640
- DataLoader check: one batch gives images (32, 3, 224, 224) and labels (32,)
- Augmentation: not applied in this stage
- Outputs: outputs\preprocessing_demo.png (before / after)

Run:

    D:\LY_Project_SLRSystem--KD_DL\venv\Scripts\python.exe preprocess.py


## 3. Prototype

- Script: prototype.py
- Flow: Camera / Image -> Preprocessing (same as above) -> Student model (MobileNetV2) -> Predicted letter
- Model: student_best_aug.pth from the base repo, plugged in as a replaceable
  component (only MobileNetV2Student and load_model() need to change for a new model)
- Webcam mode: hand goes inside the green box; screen shows prediction, model
  confidence, FPS, and a preview of the model input. Observed about 15 FPS on CPU.
  Keys: q = quit, s = save snapshot
- Image mode: pass an image path; prints original size, tensor shape, prediction, top-3

Run (webcam):

    D:\LY_Project_SLRSystem--KD_DL\venv\Scripts\python.exe prototype.py

Run (single image):

    D:\LY_Project_SLRSystem--KD_DL\venv\Scripts\python.exe prototype.py "<image path>"


## Honest baseline result (quick_check.py)

The base repo student weights were tested on 20 images per letter (A-Z) from our own
test split (520 images):

- Correct: 75 / 520 = 14.4%
- Some letters work better (C 12/20, Y 12/20, L 10/20, O 9/20), many letters got 0/20
- Frequent mistakes: H, T, W, U, V, X, J, B were mostly predicted as F or C
- Earlier baseline on 26 Kaggle test images: 7/26 (26.9%) for both teacher and student

This is the starting baseline, not our final accuracy. The base model also has only
26 outputs (A-Z), while our dataset has 29 classes.

Run:

    D:\LY_Project_SLRSystem--KD_DL\venv\Scripts\python.exe quick_check.py


## Checklist

Completed for Review II:

- [x] Dataset located, loaded and verified (29 classes, 87,000 images, balanced)
- [x] Class distribution and sample images generated
- [x] Preprocessing pipeline (resize, normalize, label encoding, split) working
- [x] Before / after preprocessing demonstration
- [x] Prototype working end to end (webcam and image mode)
- [x] Baseline of base repo weights measured honestly on our data

Intentionally left for the remaining 50%:

- [ ] Training our own teacher and student models
- [ ] Full Knowledge Distillation training
- [ ] Extending the model to our class set (29 classes)
- [ ] Dynamic letters J and Z
- [ ] Hand-landmark fusion to reduce confusion between similar signs
- [ ] Robustness testing under different lighting and camera angles
- [ ] Final accuracy improvement and comparison / benchmarking
- [ ] Final optimization and deployment