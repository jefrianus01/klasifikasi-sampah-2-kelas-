#!/usr/bin/env python3
"""
merge_and_split.py — Merge new non-recyclable images, clean, stratified re-split
===============================================================================
1. Collect ALL images (existing + new)
2. Remove corrupt, small, blurry, duplicates
3. Stratified split 70/15/15
"""

import os
import sys
import shutil
import random
import hashlib
import tempfile
from collections import defaultdict
from PIL import Image

import cv2
import numpy as np

random.seed(42)

DATASET_DIR = os.path.abspath('dataset')
TEMP_DIR = os.path.abspath('temp_nonrecyclable')
MIN_DIM = 50
BLUR_THRESHOLD = 50.0

SPLIT_RATIOS = (0.70, 0.15, 0.15)


def is_valid_image(filepath):
    try:
        img = Image.open(filepath)
        img.verify()
        img = Image.open(filepath)
        img.load()
        w, h = img.size
        if w < MIN_DIM or h < MIN_DIM:
            return False
        return True
    except Exception:
        return False


def is_blurry(filepath):
    try:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True
        return cv2.Laplacian(img, cv2.CV_64F).var() < BLUR_THRESHOLD
    except Exception:
        return True


def find_duplicates(paths):
    hash_map = defaultdict(list)
    for fpath in paths:
        try:
            with open(fpath, 'rb') as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            hash_map[md5].append(fpath)
        except Exception:
            pass
    dups = []
    for md5, plist in hash_map.items():
        if len(plist) > 1:
            dups.extend(plist[1:])
    return dups


def collect_all_images():
    """Collect ALL images from dataset + temp."""
    samples = defaultdict(list)

    # Existing dataset
    for split in ['train', 'valid', 'test']:
        for cls in ['daur_ulang', 'bukan_daur_ulang']:
            dir_path = os.path.join(DATASET_DIR, split, cls)
            if os.path.isdir(dir_path):
                for f in os.listdir(dir_path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        samples[cls].append(os.path.join(dir_path, f))

    # New non-recyclable images from temp
    hf_dir = os.path.join(TEMP_DIR, 'hf_nonrecyclable')
    if os.path.isdir(hf_dir):
        for root, dirs, files in os.walk(hf_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    samples['bukan_daur_ulang'].append(os.path.join(root, f))

    tn_dir = os.path.join(TEMP_DIR, 'trashnet_trash')
    if os.path.isdir(tn_dir):
        for f in os.listdir(tn_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                samples['bukan_daur_ulang'].append(os.path.join(tn_dir, f))

    return samples


def clean_images(samples):
    """Remove corrupt, too small, blurry, duplicate images."""
    for cls in list(samples.keys()):
        paths = samples[cls]
        print(f'\n[{cls}] Cleaning {len(paths)} images...')

        # Corrupt / too small
        valid = []
        corrupt_count = 0
        for p in paths:
            if is_valid_image(p):
                valid.append(p)
            else:
                corrupt_count += 1
        print(f'  Removed corrupt/small: {corrupt_count}')

        # Blurry
        non_blurry = []
        blurry_count = 0
        for p in valid:
            if is_blurry(p):
                blurry_count += 1
            else:
                non_blurry.append(p)
        print(f'  Removed blurry: {blurry_count}')

        # Duplicates
        dups = find_duplicates(non_blurry)
        dup_set = set(dups)
        final = [p for p in non_blurry if p not in dup_set]
        print(f'  Removed duplicates: {len(dups)}')
        print(f'  Final: {len(final)} images')

        samples[cls] = final

    return samples


def stratified_split(samples):
    X, y = [], []
    for cls in sorted(samples.keys()):
        for path in samples[cls]:
            X.append(path)
            y.append(cls)

    from sklearn.model_selection import train_test_split

    train_ratio, val_ratio, _ = SPLIT_RATIOS
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


def write_splits(splits):
    """Write splits to temp dir then swap."""
    tmp_dir = os.path.join(tempfile.gettempdir(), 'wasterecycle_merged_split')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    for split_name, items in splits.items():
        by_class = defaultdict(list)
        for path, cls in items:
            by_class[cls].append(path)

        for cls in sorted(by_class.keys()):
            cls_dir = os.path.join(tmp_dir, split_name, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for i, src_path in enumerate(by_class[cls]):
                ext = os.path.splitext(src_path)[1].lower()
                if ext not in ('.jpg', '.jpeg', '.png', '.bmp'):
                    ext = '.jpg'
                dst_path = os.path.join(cls_dir, f'{cls[:3]}_{i}{ext}')
                shutil.copy2(src_path, dst_path)

            print(f'  {split_name}/{cls}: {len(by_class[cls])} images')

    # Swap into dataset dir
    for split_name in ['train', 'valid', 'test']:
        target = os.path.join(DATASET_DIR, split_name)
        if os.path.exists(target):
            shutil.rmtree(target)
        src = os.path.join(tmp_dir, split_name)
        shutil.move(src, target)
        print(f'  Moved {split_name} -> {target}')

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)


def main():
    print('=' * 60)
    print('  Merge & Stratified Split Pipeline')
    print('=' * 60)

    print('\n[1/4] Collecting all images...')
    samples = collect_all_images()
    for cls, paths in sorted(samples.items()):
        print(f'  {cls}: {len(paths)} images')

    print('\n[2/4] Cleaning images...')
    samples = clean_images(samples)

    total = sum(len(v) for v in samples.values())
    print(f'\nTotal after cleaning: {total}')

    print('\n[3/4] Stratified split (70/15/15)...')
    splits = stratified_split(samples)
    for split_name, items in splits.items():
        by_class = defaultdict(int)
        for _, cls in items:
            by_class[cls] += 1
        print(f'  {split_name}:')
        for cls, count in sorted(by_class.items()):
            print(f'    {cls}: {count}')
        print(f'    total: {len(items)}')

    print('\n[4/4] Writing dataset...')
    write_splits(splits)

    print('\n' + '=' * 60)
    print('  MERGE & SPLIT COMPLETE!')
    print('=' * 60)

    print('\nFinal distribution:')
    for split in ['train', 'valid', 'test']:
        for cls in ['daur_ulang', 'bukan_daur_ulang']:
            d = os.path.join(DATASET_DIR, split, cls)
            c = len([f for f in os.listdir(d) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]) if os.path.isdir(d) else 0
            print(f'  {split}/{cls}: {c}')

    print(f'\nJalankan training:')
    print(f'  python train.py --backbone EfficientNetB0 --epochs-stage1 15 --epochs-stage2 30')
    print(f'  python train.py --backbone EfficientNetB0 --focal-loss --epochs-stage1 20 --epochs-stage2 40')


if __name__ == '__main__':
    main()
