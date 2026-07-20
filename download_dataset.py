#!/usr/bin/env python3
"""
download_dataset.py — Download dataset sampah dari Kaggle
===========================================================
Mendownload dataset "Recyclable and Household Waste Classification"
(15.000 gambar, 30 kategori) dari Kaggle, lalu:
  1. Memetakan ke binary class: daur_ulang / bukan_daur_ulang
  2. Split ke train (70%), valid (15%), test (15%)
  3. Menyalin gambar ke folder dataset/train, dataset/valid, dataset/test

Usage:
  python download_dataset.py
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path

random.seed(42)


# Mapping 30 kategori → binary class
RECYCLABLE_CATEGORIES = {
    # Plastic
    'Plastic water bottles', 'Plastic soda bottles', 'Plastic detergent bottles',
    'Plastic shopping bags', 'Plastic trash bags', 'Plastic food containers',
    'Plastic disposable cutlery', 'Plastic straws', 'Plastic cup lids',
    # Paper & Cardboard
    'Newspaper', 'Office paper', 'Magazines', 'Cardboard boxes', 'Cardboard packaging',
    # Glass
    'Glass beverage bottles', 'Glass food jars', 'Glass cosmetic containers',
    # Metal
    'Aluminum soda cans', 'Aluminum food cans', 'Steel food cans', 'Aerosol cans',
}

NON_RECYCLABLE_CATEGORIES = {
    # Organic waste
    'Food waste', 'Eggshells', 'Coffee grounds', 'Tea bags',
    # Textiles
    'Clothing', 'Shoes',
}


def download_kaggle_dataset(dataset_path: str, target_dir: str):
    """Download dataset from Kaggle using kagglehub."""
    import kagglehub

    print(f'Downloading dataset: {dataset_path}...')
    path = kagglehub.dataset_download(dataset_path)
    print(f'Downloaded to: {path}')

    if target_dir and os.path.exists(path):
        dest = os.path.abspath(target_dir)
        if os.path.exists(dest):
            print(f'Removing existing: {dest}')
            shutil.rmtree(dest)
        shutil.copytree(path, dest)
        print(f'Copied to: {dest}')
        return dest

    return path


def get_all_images(source_dir: str) -> list:
    """Cari semua file .png/.jpg dalam direktori (recursive)."""
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    images = []
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_exts:
                images.append(os.path.join(root, f))
    return images


def get_category_from_path(path: str, source_dir: str) -> str:
    """Ekstrak nama kategori dari path folder."""
    rel = os.path.relpath(path, source_dir)
    parts = rel.replace('\\', '/').split('/')
    return parts[0] if parts else ''


def main():
    parser = argparse.ArgumentParser(description='Download Waste Dataset from Kaggle')
    parser.add_argument('--dataset', type=str,
                        default='alistairking/recyclable-and-household-waste-classification',
                        help='Kaggle dataset identifier')
    parser.add_argument('--output', type=str, default='dataset',
                        help='Output directory for organized dataset')
    parser.add_argument('--split', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                        help='Train/valid/test split ratios')
    parser.add_argument('--max-per-class', type=int, default=0,
                        help='Max images per class (0 = all)')
    parser.add_argument('--download-dir', type=str, default='kaggle_download',
                        help='Temporary download directory')

    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    download_dir = os.path.abspath(args.download_dir)

    train_ratio, valid_ratio, test_ratio = args.split
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 0.01, \
        'Split ratios must sum to 1.0'

    # ── Step 1: Download ──
    print('=' * 60)
    print('STEP 1: Download dataset from Kaggle')
    print('=' * 60)

    data_path = download_kaggle_dataset(args.dataset, download_dir)

    images_dir = os.path.join(data_path, 'images')
    if not os.path.isdir(images_dir):
        # Try to find images folder
        for root, dirs, files in os.walk(data_path):
            if 'images' in dirs:
                images_dir = os.path.join(root, 'images')
                break
        else:
            images_dir = data_path

    # ── Step 2: Collect images per category ──
    print('')
    print('=' * 60)
    print('STEP 2: Collecting images per category')
    print('=' * 60)

    all_images = get_all_images(images_dir)
    print(f'Total images found: {len(all_images)}')

    cat_to_images = {}
    for img_path in all_images:
        cat = get_category_from_path(img_path, images_dir)
        # Handle subfolders like "Plastic water bottles/default/"
        # by keeping the top-level category name
        parts = cat.replace('\\', '/').split('/')
        cat = parts[0]

        if cat not in cat_to_images:
            cat_to_images[cat] = []
        cat_to_images[cat].append(img_path)

    print(f'Categories found: {len(cat_to_images)}')
    for cat, imgs in sorted(cat_to_images.items()):
        recyclable_label = 'daur_ulang' if cat in RECYCLABLE_CATEGORIES else \
                           'bukan_daur_ulang' if cat in NON_RECYCLABLE_CATEGORIES else \
                           'UNMAPPED'
        print(f'  [{recyclable_label:16s}] {cat:35s} -> {len(imgs)} images')

    # Check for unmapped categories
    unmapped = [c for c in cat_to_images if c not in RECYCLABLE_CATEGORIES
                and c not in NON_RECYCLABLE_CATEGORIES]
    if unmapped:
        print(f'\nWARNING: {len(unmapped)} unmapped categories: {unmapped}')
        print('These will be skipped.')

    # ── Step 3: Assign binary labels ──
    print('')
    print('=' * 60)
    print('STEP 3: Assigning binary labels')
    print('=' * 60)

    recyclable_images = []
    non_recyclable_images = []

    for cat, imgs in cat_to_images.items():
        if cat in RECYCLABLE_CATEGORIES:
            recyclable_images.extend(imgs)
        elif cat in NON_RECYCLABLE_CATEGORIES:
            non_recyclable_images.extend(imgs)

    print(f'daur_ulang (recyclable)     : {len(recyclable_images)} images')
    print(f'bukan_daur_ulang (non-recyclable): {len(non_recyclable_images)} images')

    # Limit if --max-per-class set
    if args.max_per_class > 0:
        random.shuffle(recyclable_images)
        random.shuffle(non_recyclable_images)
        recyclable_images = recyclable_images[:args.max_per_class]
        non_recyclable_images = non_recyclable_images[:args.max_per_class]
        print(f'Limited to {args.max_per_class} per class')

    # ── Step 4: Split into train/valid/test ──
    print('')
    print('=' * 60)
    print(f'STEP 4: Splitting ({train_ratio:.0%}/{valid_ratio:.0%}/{test_ratio:.0%})')
    print('=' * 60)

    def split_data(images, t_ratio, v_ratio):
        random.shuffle(images)
        n = len(images)
        n_train = int(n * t_ratio)
        n_valid = int(n * v_ratio)
        train = images[:n_train]
        valid = images[n_train:n_train + n_valid]
        test = images[n_train + n_valid:]
        return train, valid, test

    rec_train, rec_valid, rec_test = split_data(recyclable_images, train_ratio, valid_ratio)
    nonrec_train, nonrec_valid, nonrec_test = split_data(non_recyclable_images, train_ratio, valid_ratio)

    splits = {
        'train': {
            'daur_ulang': rec_train,
            'bukan_daur_ulang': nonrec_train,
        },
        'valid': {
            'daur_ulang': rec_valid,
            'bukan_daur_ulang': nonrec_valid,
        },
        'test': {
            'daur_ulang': rec_test,
            'bukan_daur_ulang': nonrec_test,
        },
    }

    # ── Step 5: Copy files ──
    print('')
    print('=' * 60)
    print('STEP 5: Copying files to organized structure')
    print('=' * 60)

    total_copied = 0
    for split_name, classes in splits.items():
        for class_name, images in classes.items():
            target_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(target_dir, exist_ok=True)

            for src_path in images:
                fname = os.path.basename(src_path)
                # Handle duplicate filenames
                dst = os.path.join(target_dir, fname)
                counter = 1
                while os.path.exists(dst):
                    name, ext = os.path.splitext(fname)
                    dst = os.path.join(target_dir, f'{name}_{counter}{ext}')
                    counter += 1
                shutil.copy2(src_path, dst)
                total_copied += 1

            print(f'  {split_name}/{class_name}: {len(images)} images')

    # ── Summary ──
    print('')
    print('=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'Total images copied: {total_copied}')
    print(f'')
    for split_name in ['train', 'valid', 'test']:
        daur = len(splits[split_name]['daur_ulang'])
        bukan = len(splits[split_name]['bukan_daur_ulang'])
        total = daur + bukan
        print(f'  {split_name}:')
        print(f'    daur_ulang     : {daur}')
        print(f'    bukan_daur_ulang: {bukan}')
        print(f'    total          : {total}')
    print(f'')
    print(f'Dataset siap di: {output_dir}')
    print(f'')
    print(f'Jalankan training:')
    print(f'  python train.py')
    print(f'  python train.py --backbone EfficientNetB0 --epochs-stage1 15 --epochs-stage2 30')


if __name__ == '__main__':
    main()
