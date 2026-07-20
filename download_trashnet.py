#!/usr/bin/env python3
"""
download_trashnet.py — Download TrashNet dataset & organize untuk WasteRecycleAI
==================================================================================
Download dataset-resized.zip (42.8 MB) dari HuggingFace TrashNet, lalu:
  - Map 6 kelas ke binary: daur_ulang (glass/paper/cardboard/plastic/metal)
                          bukan_daur_ulang (trash)
  - Split ke train/valid/test (70/15/15)
"""

import os
import sys
import zipfile
import random
import shutil
from pathlib import Path

import requests
from tqdm import tqdm

random.seed(42)

TRASHNET_URL = 'https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip'
ZIP_PATH = 'dataset-resized.zip'
EXTRACT_DIR = 'trashnet_raw'
OUTPUT_DIR = 'dataset'

RECYCLABLE = {'glass', 'paper', 'cardboard', 'plastic', 'metal'}
NON_RECYCLABLE = {'trash'}

TRAIN_RATIO, VALID_RATIO, TEST_RATIO = 0.7, 0.15, 0.15


def download_file(url, dest):
    if os.path.exists(dest):
        print(f'File already exists: {dest} ({os.path.getsize(dest) / 1024 / 1024:.1f} MB)')
        return
    print(f'Downloading {url}...')
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    with open(dest, 'wb') as f, tqdm(desc=dest, total=total, unit='B', unit_scale=True) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
    print(f'Downloaded: {dest} ({os.path.getsize(dest) / 1024 / 1024:.1f} MB)')


def extract_zip(zip_path, extract_to):
    if os.path.isdir(extract_to):
        print(f'Already extracted: {extract_to}')
        return
    print(f'Extracting {zip_path}...')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    print(f'Extracted to: {extract_to}')


def collect_images(data_dir):
    """Returns {class_name: [image_paths]}"""
    class_images = {}
    for root, dirs, files in os.walk(data_dir):
        for d in dirs:
            class_images[d] = []
        break
    for class_name in class_images:
        class_dir = os.path.join(data_dir, class_name)
        for f in os.listdir(class_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                class_images[class_name].append(os.path.join(class_dir, f))
    return class_images


def copy_images(img_list, target_dir, prefix=''):
    os.makedirs(target_dir, exist_ok=True)
    copied = 0
    for src in img_list:
        ext = os.path.splitext(src)[1]
        dst = os.path.join(target_dir, f'{prefix}{copied}{ext}')
        shutil.copy2(src, dst)
        copied += 1
    return copied


def main():
    # Step 1: Download
    download_file(TRASHNET_URL, ZIP_PATH)

    # Step 2: Extract
    extract_zip(ZIP_PATH, EXTRACT_DIR)

    # Find the actual data directory
    data_dir = EXTRACT_DIR
    for item in os.listdir(EXTRACT_DIR):
        item_path = os.path.join(EXTRACT_DIR, item)
        if os.path.isdir(item_path):
            subdirs = [d for d in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, d))]
            if subdirs:
                data_dir = item_path
                break

    print(f'Data directory: {data_dir}')

    # Step 3: Collect images per class
    class_images = collect_images(data_dir)
    print(f'\nClasses found: {list(class_images.keys())}')
    for cls, imgs in class_images.items():
        print(f'  {cls}: {len(imgs)} images')

    # Step 4: Group into recyclable / non-recyclable
    recyclable_imgs = []
    non_recyclable_imgs = []
    for cls, imgs in class_images.items():
        cls_lower = cls.lower().replace(' ', '_').replace('-', '_')
        if cls_lower in RECYCLABLE or cls in RECYCLABLE:
            recyclable_imgs.extend(imgs)
        elif cls_lower in NON_RECYCLABLE or cls in NON_RECYCLABLE:
            non_recyclable_imgs.extend(imgs)
        else:
            print(f'WARNING: Unknown class "{cls}", skipping')

    print(f'\ndaun_ulang (recyclable)     : {len(recyclable_imgs)}')
    print(f'bukan_daur_ulang (non-recyclable): {len(non_recyclable_imgs)}')

    # Step 5: Split
    random.shuffle(recyclable_imgs)
    random.shuffle(non_recyclable_imgs)

    def split(imgs):
        n = len(imgs)
        n_train = int(n * TRAIN_RATIO)
        n_valid = int(n * VALID_RATIO)
        return imgs[:n_train], imgs[n_train:n_train + n_valid], imgs[n_train + n_valid:]

    r_train, r_valid, r_test = split(recyclable_imgs)
    nr_train, nr_valid, nr_test = split(non_recyclable_imgs)

    # Step 6: Copy to organized structure
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    splits_data = {
        'train': {'daur_ulang': r_train, 'bukan_daur_ulang': nr_train},
        'valid': {'daur_ulang': r_valid, 'bukan_daur_ulang': nr_valid},
        'test':  {'daur_ulang': r_test,  'bukan_daur_ulang': nr_test},
    }

    total = 0
    print(f'\nCopying files to {OUTPUT_DIR}/...')
    for split_name, classes in splits_data.items():
        for class_name, imgs in classes.items():
            target = os.path.join(OUTPUT_DIR, split_name, class_name)
            prefix = f'{class_name[:3]}_'
            count = copy_images(imgs, target, prefix)
            total += count
            print(f'  {split_name}/{class_name}: {count} images')

    print(f'\nTotal: {total} images organized in {OUTPUT_DIR}/')
    print(f'\n  train/daur_ulang     : {len(r_train)}')
    print(f'  train/bukan_daur_ulang: {len(nr_train)}')
    print(f'  valid/daur_ulang     : {len(r_valid)}')
    print(f'  valid/bukan_daur_ulang: {len(nr_valid)}')
    print(f'  test/daur_ulang      : {len(r_test)}')
    print(f'  test/bukan_daur_ulang : {len(nr_test)}')

    # Cleanup
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
        print(f'\nCleaned up: {ZIP_PATH}')

    print(f'\nDataset siap! Jalankan training:')
    print(f'  python train.py')


if __name__ == '__main__':
    main()
