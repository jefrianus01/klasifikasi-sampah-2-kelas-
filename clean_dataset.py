#!/usr/bin/env python3
"""
clean_dataset.py — Bersihkan gambar corrupt, duplikat, dan kualitas rendah
=======================================================================
1. Hapus gambar corrupt (gagal diverifikasi PIL)
2. Hapus gambar terlalu kecil (< 50x50)
3. Hapus duplikat (berdasarkan MD5 hash)
4. Hapus gambar yang terlalu blur (Laplacian variance < threshold)
"""

import os
import sys
import hashlib
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

DATASET_DIR = os.path.abspath('dataset')
MIN_DIM = 50
BLUR_THRESHOLD = 50.0


def is_corrupt(filepath):
    try:
        img = Image.open(filepath)
        img.verify()
        img = Image.open(filepath)
        img.load()
        return False
    except Exception:
        return True


def is_too_small(filepath):
    try:
        img = Image.open(filepath)
        w, h = img.size
        return w < MIN_DIM or h < MIN_DIM
    except Exception:
        return True


def is_blurry(filepath):
    try:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        return laplacian_var < BLUR_THRESHOLD
    except Exception:
        return False


def find_duplicates(directory):
    hash_map = defaultdict(list)
    for root, dirs, files in os.walk(directory):
        for fname in files:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, 'rb') as f:
                    md5 = hashlib.md5(f.read()).hexdigest()
                hash_map[md5].append(fpath)
            except Exception:
                pass
    duplicates = []
    for md5, paths in hash_map.items():
        if len(paths) > 1:
            duplicates.extend(paths[1:])
    return duplicates


def main():
    print('=' * 60)
    print('  Dataset Cleaning Pipeline')
    print('=' * 60)

    all_images = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                all_images.append(os.path.join(root, f))

    print(f'\nTotal images: {len(all_images)}')
    print(f'Min dimension: {MIN_DIM}px')
    print(f'Blur threshold: {BLUR_THRESHOLD}')

    # Step 1: Corrupt check
    print('\n[1/4] Checking for corrupt images...')
    corrupt = []
    for fpath in all_images:
        if is_corrupt(fpath):
            corrupt.append(fpath)
    for f in corrupt:
        os.remove(f)
        print(f'  Removed (corrupt): {f}')
    print(f'  Total corrupt: {len(corrupt)}')

    # Refresh list
    all_images = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                all_images.append(os.path.join(root, f))

    # Step 2: Too small
    print('\n[2/4] Checking for images too small...')
    small = []
    for fpath in all_images:
        if is_too_small(fpath):
            small.append(fpath)
    for f in small:
        os.remove(f)
        print(f'  Removed (too small): {f}')
    print(f'  Total too small: {len(small)}')

    # Refresh list
    all_images = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                all_images.append(os.path.join(root, f))

    # Step 3: Blurry
    print('\n[3/4] Checking for blurry images...')
    blurry = []
    for fpath in all_images:
        if is_blurry(fpath):
            blurry.append(fpath)
    for f in blurry:
        os.remove(f)
        print(f'  Removed (blurry): {f}')
    print(f'  Total blurry: {len(blurry)}')

    # Refresh list
    all_images = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                all_images.append(os.path.join(root, f))

    # Step 4: Duplicates
    print('\n[4/4] Checking for duplicates...')
    duplicates = find_duplicates(DATASET_DIR)
    for f in duplicates:
        os.remove(f)
        print(f'  Removed (duplicate): {f}')
    print(f'  Total duplicates removed: {len(duplicates)}')

    # Final summary
    final_images = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                final_images.append(os.path.join(root, f))

    removed = len(all_images) - len(final_images) + len(corrupt) + len(small) + len(blurry)
    print(f'\n{"=" * 60}')
    print(f'  CLEANING COMPLETE!')
    print(f'  Before: {len(all_images)} images')
    print(f'  After:  {len(final_images)} images')
    print(f'  Removed: {removed} images (corrupt + small + blurry + dup)')

    print(f'\nFinal distribution:')
    for split in ['train', 'valid', 'test']:
        for cls in ['daur_ulang', 'bukan_daur_ulang']:
            dir_path = os.path.join(DATASET_DIR, split, cls)
            if os.path.isdir(dir_path):
                count = len([f for f in os.listdir(dir_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
                print(f'  {split}/{cls}: {count}')


if __name__ == '__main__':
    main()
