# WasteRecycleAI — Klasifikasi Sampah dengan Deep Learning

> **Proyek Akhir** — Sistem klasifikasi sampah daur ulang vs non-daur ulang menggunakan Transfer Learning dan CNN.

## 📋 Daftar Isi

- [Fitur](#fitur)
- [Struktur Proyek](#struktur-proyek)
- [Instalasi](#instalasi)
- [Dataset](#dataset)
- [Penggunaan](#penggunaan)
  - [Training](#training)
  - [Evaluasi](#evaluasi)
  - [Prediksi (Keras)](#prediksi-keras)
  - [Prediksi (ONNX)](#prediksi-onnx)
  - [Konversi ONNX](#konversi-onnx)
- [Argumen CLI](#argumen-cli)
- [Arsitektur](#arsitektur)
- [Hasil](#hasil)

---

## ✨ Fitur

| Fitur | Keterangan |
|-------|-----------|
| **Transfer Learning** | MobileNetV2, EfficientNetB0, ResNet50V2, InceptionV3, DenseNet121, NASNetMobile |
| **Fine-Tuning Bertahap** | 2 stage: head training → backbone unfreeze dengan LR rendah |
| **TensorBoard** | Logging metrik training secara real-time |
| **Class Weights** | Menangani dataset tidak seimbang otomatis |
| **Confusion Matrix** | Plot otomatis setelah training |
| **ROC Curve + AUC** | Evaluasi performa model |
| **Classification Report** | Precision, Recall, F1 per kelas |
| **History JSON** | Riwayat training disimpan ke file |
| **ONNX Export** | Ekspor otomatis model ke ONNX |
| **ONNX Runtime** | Inferensi cepat dengan ONNX (CPU/GPU) |
| **Logging Terstruktur** | Log ke file + console |
| **Mixed Precision** | Training lebih cepat dengan FP16 |

## 📁 Struktur Proyek

```
WasteRecycleAI/
├── train.py              # Training pipeline (v2.0)
├── evaluate.py           # Evaluasi model
├── predict.py            # Prediksi dengan Keras model
├── predict_onnx.py       # Prediksi dengan ONNX Runtime
├── convert_to_onnx.py    # Konversi Keras → ONNX
├── requirements.txt      # Dependencies
├── README.md             # Dokumentasi
│
├── dataset/
│   ├── train/
│   │   ├── daur_ulang/       # Gambar training (recyclable)
│   │   └── bukan_daur_ulang/ # Gambar training (non-recyclable)
│   ├── valid/
│   │   ├── daur_ulang/
│   │   └── bukan_daur_ulang/
│   └── test/
│       ├── daur_ulang/
│       └── bukan_daur_ulang/
│
├── models/               # Model terlatih (.h5, .onnx)
├── logs/                 # Log training + TensorBoard
└── results/              # Plot, metrik, classification report
```

## ⚙️ Instalasi

```bash
# 1. Clone / buka project
cd WasteRecycleAI

# 2. Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset

Dataset harus terstruktur dengan format direktori sebagai berikut:

```
dataset/
├── train/
│   ├── daur_ulang/         # Gambar sampah daur ulang
│   └── bukan_daur_ulang/   # Gambar sampah non-daur ulang
├── valid/
│   ├── daur_ulang/
│   └── bukan_daur_ulang/
└── test/
    ├── daur_ulang/
    └── bukan_daur_ulang/
```

Format gambar yang didukung: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`.

## 🚀 Penggunaan

### Training

```bash
# Training default (MobileNetV2)
python train.py

# Dengan backbone berbeda
python train.py --backbone EfficientNetB0 --epochs-stage1 15 --epochs-stage2 30

# ResNet50V2 dengan batch size lebih kecil
python train.py --backbone ResNet50V2 --batch-size 16

# Nonaktifkan class weights & skip ONNX
python train.py --no-class-weights --skip-onnx

# Mixed precision training
python train.py --mixed-precision
```

### Evaluasi

```bash
# Evaluasi model terbaru di models/
python evaluate.py

# Evaluasi model tertentu
python evaluate.py --model-path models/best_model_MobileNetV2_20250101_120000.h5

# Dengan dataset test kustom
python evaluate.py --test-dir dataset/test --batch-size 64
```

### Prediksi (Keras)

```bash
# Prediksi satu gambar
python predict.py --image path/to/image.jpg

# Prediksi banyak gambar dalam direktori
python predict.py --dir path/to/images/

# Dengan threshold berbeda
python predict.py --image image.jpg --threshold 0.7

# Model tertentu
python predict.py --image image.jpg --model models/best_model.h5
```

### Prediksi (ONNX)

```bash
# Prediksi satu gambar dengan ONNX
python predict_onnx.py --image path/to/image.jpg

# Benchmark performa (100 iterasi)
python predict_onnx.py --image image.jpg --benchmark

# GPU inference (jika onnxruntime-gpu terinstall)
python predict_onnx.py --image image.jpg --provider CUDAExecutionProvider
```

### Konversi ONNX

```bash
# Konversi model terbaru
python convert_to_onnx.py

# Model tertentu dengan opset 15
python convert_to_onnx.py --model models/my_model.h5 --opset 15

# Konversi + kuantisasi (uint8)
python convert_to_onnx.py --quantize
```

## 🧠 Arsitektur

```
Input (224x224x3)
    │
    ▼
Backbone (Pre-trained ImageNet)
- MobileNetV2 / EfficientNetB0 / ResNet50V2 / dll
    │
    ▼
GlobalAveragePooling2D
    │
    ▼
BatchNormalization → Dropout (0.3)
    │
    ▼
Dense (128) → ReLU → BatchNorm → Dropout (0.3)
    │
    ▼
Output (1) → Sigmoid → [daur_ulang / bukan_daur_ulang]
```

### Stage Training

| Stage | Backbone | Learning Rate | Epochs |
|-------|----------|--------------|--------|
| 1 | **Frozen** | 1×10⁻³ | 10 |
| 2 | **Unfrozen** (BN tetap freeze) | 1×10⁻⁵ | 20 |

## 📈 Hasil

Setelah training, hasil akan tersimpan di:

| File | Path |
|------|------|
| Best model (H5) | `models/best_model_{backbone}_{timestamp}.h5` |
| Best model (ONNX) | `models/best_model_{backbone}_{timestamp}.onnx` |
| Final model (H5) | `models/final_model_{backbone}_{timestamp}.h5` |
| Training history plot | `results/training_history_{run}.png` |
| Training history JSON | `results/training_history_{run}.json` |
| Confusion matrix | `results/confusion_matrix_{run}.png` |
| ROC curve | `results/roc_curve_{run}.png` |
| Classification report | `results/classification_report_{run}.json` |
| Metrics summary | `results/metrics_{run}.json` |
| Training log | `logs/training_*.log` |
| TensorBoard log | `logs/tensorboard/{run}/` |

### TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

---

_Dibuat untuk Proyek Akhir — Klasifikasi Sampah dengan Deep Learning_
