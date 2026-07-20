#!/usr/bin/env python3
"""
evaluate.py — Evaluasi model WasteRecycleAI
=============================================
Memuat model terlatih (.h5) dan mengevaluasi pada test set.

Output:
  - Classification Report (console + JSON)
  - Confusion Matrix plot
  - ROC Curve plot
  - Metrics summary (accuracy, precision, recall, f1, auc)
"""

import os
import sys
import json
import logging
import argparse
import datetime
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

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

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────

def setup_logging(results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    logger = logging.getLogger('WasteRecycleAI_Eval')
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(
        os.path.join(results_dir, f'evaluate_{timestamp}.log'), encoding='utf-8',
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ──────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────

PREPROCESS_MAP = {
    'MobileNetV2': mobilenetv2_preprocess,
    'EfficientNetB0': efficientnet_preprocess,
    'ResNet50V2': resnetv2_preprocess,
    'InceptionV3': inceptionv3_preprocess,
    'DenseNet121': densenet_preprocess,
    'NASNetMobile': nasnet_preprocess,
}


def load_test_data(test_dir: str, img_size: int, batch_size: int, backbone: str = 'EfficientNetB0'):
    preprocess_fn = PREPROCESS_MAP[backbone]
    datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_fn,
    )

    test_gen = datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
    )

    return test_gen


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate(model, test_gen, logger, results_dir: str, run_name: str):
    logger.info('Running evaluation on test set...')
    logger.info(f'Test samples: {test_gen.samples}')

    test_gen.reset()
    y_pred_prob = model.predict(test_gen, verbose=1).flatten()
    y_pred = (y_pred_prob > 0.5).astype(np.int64)
    y_true = test_gen.classes[:len(y_pred)]

    class_labels = sorted(test_gen.class_indices, key=test_gen.class_indices.get)

    report_dict = classification_report(
        y_true, y_pred, target_names=class_labels, output_dict=True,
    )
    report_str = classification_report(
        y_true, y_pred, target_names=class_labels,
    )

    logger.info(f'\nClassification Report:\n{report_str}')

    report_path = os.path.join(results_dir, f'classification_report_{run_name}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2)
    logger.info(f'Saved: {report_path}')

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_labels, yticklabels=class_labels,
        annot_kws={'size': 14},
    )
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(results_dir, f'confusion_matrix_{run_name}.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Saved: {cm_path}')

    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr, tpr, color='darkorange', lw=2,
        label=f'ROC (AUC = {roc_auc:.4f})',
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(results_dir, f'roc_curve_{run_name}.png')
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Saved: {roc_path} (AUC = {roc_auc:.4f})')

    test_loss, test_acc = model.evaluate(test_gen, verbose=0)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary',
    )

    logger.info('─' * 50)
    logger.info('EVALUATION SUMMARY')
    logger.info('─' * 50)
    logger.info(f'  Accuracy : {test_acc:.4f}')
    logger.info(f'  Loss     : {test_loss:.4f}')
    logger.info(f'  Precision: {precision:.4f}')
    logger.info(f'  Recall   : {recall:.4f}')
    logger.info(f'  F1-Score : {f1:.4f}')
    logger.info(f'  AUC      : {roc_auc:.4f}')
    logger.info('─' * 50)

    metrics = {
        'accuracy': float(test_acc),
        'loss': float(test_loss),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
    }
    metrics_path = os.path.join(results_dir, f'metrics_{run_name}.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f'Saved: {metrics_path}')


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='WasteRecycleAI — Evaluate Model')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to .h5 model file')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory containing model files')
    parser.add_argument('--test-dir', type=str, default=os.path.join('dataset', 'test'),
                        help='Test dataset directory')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument(
        '--backbone', type=str, default='EfficientNetB0',
        choices=list(PREPROCESS_MAP.keys()),
        help='Backbone used during training (for preprocessing)',
    )

    args = parser.parse_args()

    results_dir = args.results_dir
    logger = setup_logging(results_dir)

    logger.info('WasteRecycleAI — Evaluation')
    logger.info(f'Results dir: {results_dir}')

    if args.model_path:
        model_path = args.model_path
    elif args.model_dir:
        h5_files = sorted([
            os.path.join(args.model_dir, f)
            for f in os.listdir(args.model_dir)
            if f.endswith('.h5')
        ])
        if not h5_files:
            logger.error(f'No .h5 files found in {args.model_dir}')
            sys.exit(1)
        model_path = h5_files[-1]
        logger.info(f'Auto-selected model: {model_path}')
    else:
        logger.error('Provide --model-path or --model-dir')
        sys.exit(1)

    if not os.path.exists(model_path):
        logger.error(f'Model not found: {model_path}')
        sys.exit(1)

    model = keras.models.load_model(model_path)
    logger.info(f'Model loaded: {model_path}')
    logger.info(f'Model name: {model.name}')

    test_gen = load_test_data(args.test_dir, args.img_size, args.batch_size, args.backbone)
    if test_gen.samples == 0:
        logger.error(f'No images found in {args.test_dir}')
        sys.exit(1)

    run_name = os.path.splitext(os.path.basename(model_path))[0]
    evaluate(model, test_gen, logger, results_dir, run_name)

    logger.info('Evaluation complete!')


if __name__ == '__main__':
    main()
