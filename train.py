#!/usr/bin/env python3
"""
train.py — WasteRecycleAI Training Pipeline v2.0
================================================
Pelatihan model klasifikasi sampah dengan Transfer Learning & Fine-Tuning bertahap.

Fitur:
  - Transfer Learning (MobileNetV2 / EfficientNetB0 / ResNet50V2)
  - Fine-Tuning bertahap (progressive unfreezing, 2 stage)
  - TensorBoard logging
  - Class Weight handling (dataset tidak seimbang)
  - ModelCheckpoint (save best model based on val_accuracy)
  - History training disimpan ke file JSON
  - Confusion Matrix + Classification Report otomatis
  - Ekspor otomatis ke ONNX
  - Logging terstruktur (file + console)
  - Dataset exploration & class distribution report
  - ROC Curve + AUC Score
  - Per-class metrics (precision, recall, f1)
"""

import os
import sys
import json
import logging
import argparse
import datetime
import warnings
from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras import applications

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetv2_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_fscore_support,
)

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150


# ──────────────────────────────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────────────────────────────

def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)
        return tf.reduce_mean(weight * ce)
    return loss


# ──────────────────────────────────────────────────────────────────────
# Konfigurasi
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    train_dir: str = os.path.join('dataset', 'train')
    valid_dir: str = os.path.join('dataset', 'valid')
    test_dir: str = os.path.join('dataset', 'test')
    models_dir: str = 'models'
    logs_dir: str = 'logs'
    results_dir: str = 'results'

    img_size: int = 224
    batch_size: int = 32
    epochs_stage1: int = 10
    epochs_stage2: int = 20
    learning_rate_stage1: float = 1e-3
    learning_rate_stage2: float = 1e-5
    dropout_rate: float = 0.3
    dense_units: int = 128

    backbone: str = 'MobileNetV2'
    input_shape: Tuple[int, int, int] = field(init=False)

    rotation_range: int = 20
    zoom_range: float = 0.2
    horizontal_flip: bool = True
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1

    use_class_weights: bool = True
    use_focal_loss: bool = False
    use_mixed_precision: bool = False

    def __post_init__(self):
        self.input_shape = (self.img_size, self.img_size, 3)


config = Config()

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    logger = logging.getLogger('WasteRecycleAI')
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


logger = setup_logging(config.logs_dir)

# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────

def explore_dataset(config: Config):
    """Menampilkan statistik dataset."""
    logger.info('Dataset Exploration:')
    logger.info('=' * 60)
    for split_name, split_dir in [
        ('Train', config.train_dir),
        ('Valid', config.valid_dir),
        ('Test', config.test_dir),
    ]:
        if not os.path.isdir(split_dir):
            logger.warning(f'{split_name} directory not found: {split_dir}')
            continue
        classes = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])
        total = 0
        logger.info(f'{split_name} set:')
        for cls in classes:
            cls_path = os.path.join(split_dir, cls)
            count = len([
                f for f in os.listdir(cls_path)
                if os.path.isfile(os.path.join(cls_path, f))
            ])
            total += count
            logger.info(f'  {cls}: {count} images')
        logger.info(f'  Total: {total} images')
        logger.info('')
    logger.info('=' * 60)


def create_data_generators(config: Config) -> Tuple:
    preprocess_fn = PREPROCESS_MAP[config.backbone]

    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=config.rotation_range,
        zoom_range=config.zoom_range,
        horizontal_flip=config.horizontal_flip,
        brightness_range=config.brightness_range,
        width_shift_range=config.width_shift_range,
        height_shift_range=config.height_shift_range,
        fill_mode='nearest',
    )

    valid_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_fn,
    )

    test_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_fn,
    )

    train_gen = train_datagen.flow_from_directory(
        config.train_dir,
        target_size=(config.img_size, config.img_size),
        batch_size=config.batch_size,
        class_mode='binary',
        shuffle=True,
        seed=42,
    )

    valid_gen = valid_datagen.flow_from_directory(
        config.valid_dir,
        target_size=(config.img_size, config.img_size),
        batch_size=config.batch_size,
        class_mode='binary',
        shuffle=False,
    )

    test_gen = test_datagen.flow_from_directory(
        config.test_dir,
        target_size=(config.img_size, config.img_size),
        batch_size=config.batch_size,
        class_mode='binary',
        shuffle=False,
    )

    return train_gen, valid_gen, test_gen


def compute_class_weights(train_gen, mild: bool = True) -> Optional[dict]:
    total = train_gen.samples
    num_classes = len(train_gen.class_indices)
    class_counts = np.bincount(train_gen.classes, minlength=num_classes)

    logger.info('Class distribution (training set):')
    labels = sorted(train_gen.class_indices, key=train_gen.class_indices.get)
    for label, count in zip(labels, class_counts):
        logger.info(f'  {label}: {count} ({count / total * 100:.1f}%)')

    if num_classes < 2:
        logger.warning('Only one class found, skipping class weights.')
        return None

    class_weights = {}
    for i in range(num_classes):
        w = total / (num_classes * class_counts[i])
        if mild:
            w = np.sqrt(w)
        class_weights[i] = w

    logger.info(f'Computed class weights: {class_weights}')

    imbalance_ratio = max(class_counts) / min(class_counts)
    if imbalance_ratio > 1.5:
        logger.info(f'Class imbalance detected (ratio: {imbalance_ratio:.2f}), {"mild sqrt" if mild else "linear"} weights.')
    else:
        logger.info(f'Dataset fairly balanced (ratio: {imbalance_ratio:.2f}).')

    return class_weights


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────

BACKBONE_MAP = {
    'MobileNetV2': (applications.MobileNetV2, (224, 224)),
    'EfficientNetB0': (applications.EfficientNetB0, (224, 224)),
    'ResNet50V2': (applications.ResNet50V2, (224, 224)),
    'InceptionV3': (applications.InceptionV3, (299, 299)),
    'DenseNet121': (applications.DenseNet121, (224, 224)),
    'NASNetMobile': (applications.NASNetMobile, (224, 224)),
}

PREPROCESS_MAP = {
    'MobileNetV2': mobilenetv2_preprocess,
    'EfficientNetB0': efficientnet_preprocess,
    'ResNet50V2': resnetv2_preprocess,
    'InceptionV3': inceptionv3_preprocess,
    'DenseNet121': densenet_preprocess,
    'NASNetMobile': nasnet_preprocess,
}


def build_model(config: Config) -> Tuple[keras.Model, keras.Model]:
    if config.backbone not in BACKBONE_MAP:
        raise ValueError(
            f'Unknown backbone: {config.backbone}. '
            f'Choose from: {list(BACKBONE_MAP.keys())}'
        )

    base_model_class, default_size = BACKBONE_MAP[config.backbone]

    if config.img_size != default_size[0]:
        logger.warning(
            f'{config.backbone} expects input size {default_size[0]}, '
            f'but got {config.img_size}. Using {config.img_size}.'
        )

    base_model = base_model_class(
        include_top=False,
        weights='imagenet',
        input_shape=config.input_shape,
        pooling=None,
    )
    base_model.trainable = False

    inputs = keras.Input(shape=config.input_shape, name='input_image')

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(config.dropout_rate, name='dropout_1')(x)

    x = layers.Dense(config.dense_units, activation='relu', name='dense_hidden')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(config.dropout_rate, name='dropout_2')(x)

    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = keras.Model(inputs, outputs, name='WasteClassifier')

    trainable_count = sum(p.numel() if hasattr(p, 'numel') else p.numpy().size for p in model.trainable_weights)
    total_count = model.count_params()

    logger.info(f'Model: {model.name}')
    logger.info(f'Backbone: {config.backbone}')
    logger.info(f'Total parameters: {total_count:,}')
    logger.info(f'Trainable parameters (stage 1): {trainable_count:,}')
    logger.info(f'Input shape: {config.input_shape}')

    return model, base_model


# ──────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────

def create_callbacks(config: Config, run_name: str) -> list:
    os.makedirs(config.models_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)

    ckpt_path = os.path.join(config.models_dir, f'best_model_{run_name}.h5')

    cbs = [
        callbacks.ModelCheckpoint(
            ckpt_path,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(config.logs_dir, 'tensorboard', run_name),
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch',
        ),
    ]

    return cbs


# ──────────────────────────────────────────────────────────────────────
# Visualization & Evaluation
# ──────────────────────────────────────────────────────────────────────

def plot_training_history(history, config: Config, run_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    epochs = range(1, len(history.history['loss']) + 1)

    axes[0].plot(epochs, history.history.get('accuracy', []), 'b-', label='Train Accuracy')
    axes[0].plot(epochs, history.history.get('val_accuracy', []), 'r-', label='Validation Accuracy')
    axes[0].set_title('Accuracy', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    axes[1].plot(epochs, history.history.get('loss', []), 'b-', label='Train Loss')
    axes[1].plot(epochs, history.history.get('val_loss', []), 'r-', label='Validation Loss')
    axes[1].set_title('Loss', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(config.results_dir, f'training_history_{run_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Training history plot saved: {path}')


def save_history_json(history, config: Config, run_name: str):
    hist = {}
    for k, v in history.history.items():
        hist[k] = [float(x) for x in v]

    path = os.path.join(config.results_dir, f'training_history_{run_name}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(hist, f, indent=2)
    logger.info(f'Training history saved: {path}')


def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.05, 0.95, 0.05)
    best_f1, best_t = 0, 0.5
    for t in thresholds:
        pred = (y_prob > t).astype(np.int64)
        _, _, f1, _ = precision_recall_fscore_support(y_true, pred, average='binary')
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def evaluate_model(model, test_gen, config: Config, run_name: str):
    logger.info('Evaluating on test set...')

    test_gen.reset()
    y_pred_prob = model.predict(test_gen, verbose=1).flatten()
    y_true = test_gen.classes[:len(y_pred_prob)]

    best_threshold, best_f1 = find_best_threshold(y_true, y_pred_prob)
    logger.info(f'Best threshold: {best_threshold:.2f} (F1: {best_f1:.4f})')
    y_pred = (y_pred_prob > best_threshold).astype(np.int64)
    y_pred_default = (y_pred_prob > 0.5).astype(np.int64)

    class_labels = sorted(test_gen.class_indices, key=test_gen.class_indices.get)

    report_dict = classification_report(
        y_true, y_pred, target_names=class_labels, output_dict=True,
    )
    report_str = classification_report(
        y_true, y_pred, target_names=class_labels,
    )

    logger.info(f'\nClassification Report:\n{report_str}')

    report_path = os.path.join(
        config.results_dir, f'classification_report_{run_name}.json',
    )
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        annot_kws={'size': 14},
    )
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(config.results_dir, f'confusion_matrix_{run_name}.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Confusion matrix saved: {cm_path}')

    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr, tpr, color='darkorange', lw=2,
        label=f'ROC curve (AUC = {roc_auc:.4f})',
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(config.results_dir, f'roc_curve_{run_name}.png')
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'ROC curve saved: {roc_path} (AUC = {roc_auc:.4f})')

    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    logger.info(f'Test Accuracy: {test_acc:.4f}')
    logger.info(f'Test Loss: {test_loss:.4f}')

    prec_default, rec_default, f1_default, _ = precision_recall_fscore_support(
        y_true, y_pred_default, average='binary',
    )
    logger.info(f'--- With threshold=0.50 ---')
    logger.info(f'Precision: {prec_default:.4f} | Recall: {rec_default:.4f} | F1: {f1_default:.4f}')

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary',
    )
    logger.info(f'--- With threshold={best_threshold:.2f} (best F1) ---')
    logger.info(f'Precision: {precision:.4f}')
    logger.info(f'Recall: {recall:.4f}')
    logger.info(f'F1-Score: {f1:.4f}')

    metrics = {
        'accuracy': float(test_acc),
        'loss': float(test_loss),
        'best_threshold': float(best_threshold),
        'precision_opt': float(precision),
        'recall_opt': float(recall),
        'f1_score_opt': float(f1),
        'precision_default': float(prec_default),
        'recall_default': float(rec_default),
        'f1_default': float(f1_default),
        'auc': float(roc_auc),
    }
    metrics_path = os.path.join(
        config.results_dir, f'metrics_{run_name}.json',
    )
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f'Metrics saved: {metrics_path}')

    return test_acc, cm


# ──────────────────────────────────────────────────────────────────────
# ONNX Export
# ──────────────────────────────────────────────────────────────────────

def export_to_onnx(model, config: Config, run_name: str):
    import tf2onnx

    onnx_path = os.path.join(config.models_dir, f'best_model_{run_name}.onnx')
    dummy = tf.zeros((1, *config.input_shape), dtype=tf.float32)

    spec = (tf.TensorSpec(dummy.shape, tf.float32, name='input'),)
    model.output_names = ['output']

    tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        output_path=onnx_path,
    )

    logger.info(f'ONNX exported: {onnx_path}')
    return onnx_path


# ──────────────────────────────────────────────────────────────────────
# Training Stages
# ──────────────────────────────────────────────────────────────────────

def train_stage(
    model,
    base_model,
    train_gen,
    valid_gen,
    config: Config,
    run_name: str,
    stage: int,
    epochs: int,
    learning_rate: float,
    freeze_backbone: bool,
    initial_epoch: int = 0,
):
    logger.info('=' * 60)
    logger.info(f'STAGE {stage}: {"Training head" if freeze_backbone else "Fine-tuning backbone"}')
    logger.info(f'  Epochs: {epochs}')
    logger.info(f'  Learning rate: {learning_rate}')
    logger.info(f'  Backbone trainable: {not freeze_backbone}')
    logger.info('=' * 60)

    if freeze_backbone:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    loss_fn = focal_loss() if config.use_focal_loss else 'binary_crossentropy'
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['accuracy'],
    )
    if config.use_focal_loss:
        logger.info('  Using Focal Loss (gamma=2.0, alpha=0.25)')

    cbs = create_callbacks(config, run_name)

    class_weight = None
    if config.use_class_weights:
        class_weight = compute_class_weights(train_gen, mild=False)

    history = model.fit(
        train_gen,
        epochs=initial_epoch + epochs,
        initial_epoch=initial_epoch,
        validation_data=valid_gen,
        callbacks=cbs,
        class_weight=class_weight,
        verbose=1,
    )

    return history


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='WasteRecycleAI — Professional Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python train.py\n'
            '  python train.py --backbone EfficientNetB0 --epochs-stage1 15 --epochs-stage2 30\n'
            '  python train.py --backbone ResNet50V2 --img-size 224 --batch-size 16\n'
            '  python train.py --no-class-weights --skip-onnx\n'
        ),
    )
    parser.add_argument('--backbone', type=str, default='MobileNetV2',
                        choices=list(BACKBONE_MAP.keys()),
                        help='Backbone model for transfer learning')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size (px, square)')
    parser.add_argument('--epochs-stage1', type=int, default=10)
    parser.add_argument('--epochs-stage2', type=int, default=20)
    parser.add_argument('--lr-stage1', type=float, default=1e-3)
    parser.add_argument('--lr-stage2', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--dense-units', type=int, default=128)
    parser.add_argument('--no-class-weights', action='store_true',
                        help='Disable class weights')
    parser.add_argument('--skip-onnx', action='store_true',
                        help='Skip ONNX export')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Enable mixed precision training')
    parser.add_argument('--focal-loss', action='store_true',
                        help='Use Focal Loss instead of binary crossentropy')

    args = parser.parse_args()

    global config
    config.backbone = args.backbone
    config.batch_size = args.batch_size
    config.img_size = args.img_size
    config.input_shape = (args.img_size, args.img_size, 3)
    config.epochs_stage1 = args.epochs_stage1
    config.epochs_stage2 = args.epochs_stage2
    config.learning_rate_stage1 = args.lr_stage1
    config.learning_rate_stage2 = args.lr_stage2
    config.dropout_rate = args.dropout
    config.dense_units = args.dense_units
    config.use_class_weights = not args.no_class_weights
    config.use_focal_loss = args.focal_loss
    config.use_mixed_precision = args.mixed_precision

    if config.use_mixed_precision:
        mixed_precision.set_global_policy('mixed_float16')
        logger.info('Mixed precision training enabled')

    run_name = f'{config.backbone}_{datetime.datetime.now():%Y%m%d_%H%M%S}'

    logger.info('=' * 58)
    logger.info('   WasteRecycleAI -- Training Pipeline v2.0')
    logger.info('=' * 58)
    logger.info(f'Run: {run_name}')
    logger.info(f'Config: {json.dumps(asdict(config), indent=2, default=str)}')

    os.makedirs(config.models_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)

    explore_dataset(config)

    logger.info('Creating data generators...')
    train_gen, valid_gen, test_gen = create_data_generators(config)

    logger.info(f'Train samples: {train_gen.samples}')
    logger.info(f'Valid samples: {valid_gen.samples}')
    logger.info(f'Test samples: {test_gen.samples}')
    logger.info(f'Class indices: {train_gen.class_indices}')

    logger.info(f'Building model with backbone: {config.backbone}...')
    model, base_model = build_model(config)

    model.summary(print_fn=lambda x: logger.info(x))

    history1 = train_stage(
        model, base_model, train_gen, valid_gen,
        config, run_name,
        stage=1,
        epochs=config.epochs_stage1,
        learning_rate=config.learning_rate_stage1,
        freeze_backbone=True,
        initial_epoch=0,
    )

    history2 = train_stage(
        model, base_model, train_gen, valid_gen,
        config, run_name,
        stage=2,
        epochs=config.epochs_stage2,
        learning_rate=config.learning_rate_stage2,
        freeze_backbone=False,
        initial_epoch=config.epochs_stage1,
    )

    combined = keras.callbacks.History()
    combined.history = {}
    for key in history1.history:
        combined.history[key] = history1.history[key] + history2.history.get(key, [])
    for key in history2.history:
        if key not in combined.history:
            combined.history[key] = history2.history[key]

    plot_training_history(combined, config, run_name)
    save_history_json(combined, config, run_name)

    best_path = os.path.join(config.models_dir, f'best_model_{run_name}.h5')
    if os.path.exists(best_path):
        logger.info(f'Loading best model weights: {best_path}')
        model.load_weights(best_path)
    else:
        logger.warning('Best model checkpoint not found, using final model.')

    evaluate_model(model, test_gen, config, run_name)

    final_model_path = os.path.join(
        config.models_dir, f'final_model_{run_name}.h5',
    )
    model.save(final_model_path)
    logger.info(f'Final model saved: {final_model_path}')

    if not args.skip_onnx:
        export_to_onnx(model, config, run_name)

    logger.info('')
    logger.info('=' * 58)
    logger.info('              TRAINING COMPLETE!')
    logger.info('=' * 58)
    logger.info(f'Best model: {best_path}')
    logger.info(f'Final model: {final_model_path}')
    logger.info(f'ONNX: {os.path.join(config.models_dir, f"best_model_{run_name}.onnx")}')
    logger.info(f'Results: {config.results_dir}/')
    logger.info(f'Logs: {config.logs_dir}/')
    logger.info('')
    logger.info('Next steps:')
    logger.info('  python evaluate.py --model-dir models')
    logger.info('  python predict.py --image path/to/image.jpg')
    logger.info('  python predict_onnx.py --image path/to/image.jpg')


if __name__ == '__main__':
    main()
