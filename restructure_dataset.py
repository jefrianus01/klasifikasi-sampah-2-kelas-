#!/usr/bin/env python3
"""
restructure_dataset.py — Stratified split dataset
Menggabung train/valid/test → stratified split ulang 70/15/15
"""

import os
import shutil
import random
import tempfile
from collections import defaultdict
from sklearn.model_selection import train_test_split

random.seed(42)
SPLIT = (0.70, 0.15, 0.15)

ORIG_DIRS = ['dataset/train', 'dataset/valid', 'dataset/test']
ORIG_TRAIN = 'dataset/train'
OUT_DIR = 'dataset'

def collect_samples():
    # Detect classes from ALL existing split dirs
    all_classes = set()
    for split_dir in ORIG_DIRS:
        if os.path.isdir(split_dir):
            for d in os.listdir(split_dir):
                if os.path.isdir(os.path.join(split_dir, d)):
                    all_classes.add(d)
    classes = sorted(all_classes)
    print(f'  Detected classes: {classes}')

    samples = defaultdict(list)
    for split_dir in ORIG_DIRS:
        if not os.path.isdir(split_dir):
            print(f'  WARNING: {split_dir} not found, skipping')
            continue
        for cls in classes:
            cls_path = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_path):
                print(f'  WARNING: {cls_path} not found, skipping')
                continue
            for fname in os.listdir(cls_path):
                src = os.path.normpath(os.path.join(cls_path, fname))
                if os.path.isfile(src):
                    samples[cls].append(src)
    return classes, samples

def stratified_split(samples_dict, classes, ratios):
    train_ratio, val_ratio, _ = ratios
    X, y = [], []
    for cls in classes:
        for path in samples_dict[cls]:
            X.append(path)
            y.append(cls)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), stratify=y, random_state=42
    )

    val_size = val_ratio / (1 - train_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_size), stratify=y_temp, random_state=42
    )

    splits = {
        'train': list(zip(X_train, y_train)),
        'valid': list(zip(X_val, y_val)),
        'test': list(zip(X_test, y_test)),
    }
    return splits

def write_splits_to_temp(splits):
    """Write to temp directory first, to avoid corrupting source data."""
    tmp_dir = os.path.join(tempfile.gettempdir(), 'wasterecycle_dataset_split')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    for split_name, items in splits.items():
        by_class = defaultdict(list)
        for path, cls in items:
            by_class[cls].append(path)

        total = 0
        for cls in sorted(by_class.keys()):
            cls_dir = os.path.join(tmp_dir, split_name, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for src_path in by_class[cls]:
                fname = os.path.basename(src_path)
                dst_path = os.path.join(cls_dir, fname)
                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(fname)
                    dst_path = os.path.join(cls_dir, f'{base}_{random.randint(1000,9999)}{ext}')
                shutil.copy2(src_path, dst_path)
                total += 1
            print(f'  {cls}: {len(by_class[cls])} images')
        print(f'  Total {split_name}: {total} images\n')

    return tmp_dir

def swap_dirs(tmp_dir):
    """Move temp dirs into place."""
    for split_name in ['train', 'valid', 'test']:
        target = os.path.join(OUT_DIR, split_name)
        if os.path.exists(target):
            shutil.rmtree(target)
        src = os.path.join(tmp_dir, split_name)
        shutil.move(src, target)
        print(f'  Moved {split_name} -> {target}')

def main():
    print('=' * 60)
    print('  Restructuring dataset with stratified split')
    print('=' * 60)
    print('\nCollecting samples...')
    classes, samples = collect_samples()
    for cls in classes:
        print(f'  {cls}: {len(samples[cls])} total samples')
    total = sum(len(v) for v in samples.values())
    print(f'  Total: {total} samples')

    print(f'\nPerforming stratified split ({SPLIT[0]*100:.0f}/{SPLIT[1]*100:.0f}/{SPLIT[2]*100:.0f})...')
    splits = stratified_split(samples, classes, SPLIT)

    print('\nWriting to temp directory...')
    tmp_dir = write_splits_to_temp(splits)

    print('Swapping directories...')
    swap_dirs(tmp_dir)

    print('=' * 60)
    print('  Dataset restructuring complete!')
    print('=' * 60)
    print('\nNew dataset distribution:')
    for split_name in ['train', 'valid', 'test']:
        split_path = os.path.join(OUT_DIR, split_name)
        print(f'\n{split_name}:')
        for cls in sorted(os.listdir(split_path)):
            count = len(os.listdir(os.path.join(split_path, cls)))
            print(f'  {cls}: {count}')

if __name__ == '__main__':
    main()
