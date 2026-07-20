#!/usr/bin/env python3
"""
download_more_nonrecyclable.py
Download more non-recyclable waste images to reach minimum 1000 images.

Sources:
  1. HuggingFace: omasteam/waste-garbage-management-dataset (biological, clothes, shoes, trash, battery)
  2. TrashNet trash class

Usage:
  python download_more_nonrecyclable.py
"""

import os
import shutil
import random
import hashlib
import zipfile

import requests
from PIL import Image

random.seed(42)

OUTPUT_DIR = os.path.abspath('dataset')
TEMP_DIR = os.path.abspath('temp_nonrecyclable')
TRASHNET_ZIP = 'dataset-resized.zip'
TRASHNET_URL = 'https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip'

HF_DATASET = 'omasteam/waste-garbage-management-dataset'
# Non-recyclable categories from the HF dataset
NON_RECYCLABLE_CATEGORIES = ['biological', 'clothes', 'shoes', 'trash', 'battery']

MAX_IMAGES = 2000  # Max total non-recyclable to download (we need ~863+ more)


def is_valid_image(filepath):
    try:
        img = Image.open(filepath)
        img.verify()
        img = Image.open(filepath)
        w, h = img.size
        return w >= 100 and h >= 100
    except:
        return False


def download_file(url, save_path, timeout=30):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(resp.content)
        return True
    except Exception as e:
        return False


def download_hf_dataset():
    """Download non-recyclable categories from HuggingFace dataset using snapshot_download."""
    print('\n' + '=' * 60)
    print('Downloading from HuggingFace: omasteam/waste-garbage-management-dataset')
    print('=' * 60)

    from huggingface_hub import snapshot_download

    out_dir = os.path.join(TEMP_DIR, 'hf_nonrecyclable')

    # Build allow patterns for non-recyclable categories
    allow_patterns = []
    for cat in NON_RECYCLABLE_CATEGORIES:
        allow_patterns.append(f'{cat}/*')

    print(f'Downloading categories: {", ".join(NON_RECYCLABLE_CATEGORIES)}')
    print('Using snapshot_download (git LFS, much faster)...')

    snapshot_download(
        repo_id=HF_DATASET,
        repo_type='dataset',
        local_dir=out_dir,
        allow_patterns=allow_patterns,
        max_workers=8,
    )

    # Count downloaded images per category
    total = 0
    for cat in NON_RECYCLABLE_CATEGORIES:
        cat_dir = os.path.join(out_dir, cat)
        if os.path.isdir(cat_dir):
            count = len([f for f in os.listdir(cat_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            total += count
            print(f'  {cat}: {count} images downloaded')

    print(f'\nTotal downloaded from HF: {total}')
    return out_dir


def download_trashnet_trash():
    """Extract TrashNet trash images."""
    print('\n' + '=' * 60)
    print('Extracting TrashNet trash images...')
    print('=' * 60)

    if not os.path.exists(TRASHNET_ZIP):
        print(f'Downloading TrashNet...')
        resp = requests.get(TRASHNET_URL, stream=True)
        resp.raise_for_status()
        with open(TRASHNET_ZIP, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print('Downloaded TrashNet')

    out_dir = os.path.join(TEMP_DIR, 'trashnet_trash')
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    with zipfile.ZipFile(TRASHNET_ZIP, 'r') as zf:
        for member in zf.namelist():
            if '/trash/' in member and member.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                zf.extract(member, TEMP_DIR)
                src = os.path.join(TEMP_DIR, member)
                dst = os.path.join(out_dir, f'trashnet_{os.path.basename(member)}')
                if os.path.exists(src):
                    shutil.move(src, dst)
                    count += 1

    print(f'Extracted {count} TrashNet trash images')
    return out_dir


def deduplicate_by_hash(directory):
    """Remove duplicate images based on MD5 hash."""
    hashes = {}
    duplicates = 0
    for root, dirs, files in os.walk(directory):
        for fname in files:
            fpath = os.path.join(root, fname)
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            with open(fpath, 'rb') as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            if md5 in hashes:
                os.remove(fpath)
                duplicates += 1
            else:
                hashes[md5] = fpath
    return duplicates


def main():
    print('=' * 60)
    print('Download More Non-Recyclable Waste Images')
    print('=' * 60)

    # Count current state
    total_nonrec = 0
    total_rec = 0
    for split in ['train', 'valid', 'test']:
        for cls in ['daur_ulang', 'bukan_daur_ulang']:
            dir_path = os.path.join(OUTPUT_DIR, split, cls)
            if os.path.isdir(dir_path):
                c = len([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                if cls == 'daur_ulang':
                    total_rec += c
                else:
                    total_nonrec += c

    print(f'\nCurrent state:')
    print(f'  daur_ulang     : {total_rec}')
    print(f'  bukan_daur_ulang: {total_nonrec}')
    needed = max(0, 1000 - total_nonrec)
    print(f'  Need {needed} more bukan_daur_ulang images')

    if needed <= 0:
        print('\nTarget already achieved!')
        return

    os.makedirs(TEMP_DIR, exist_ok=True)

    # Step 1: Download from HuggingFace dataset
    hf_dir = download_hf_dataset()

    # Step 2: Download TrashNet trash
    trashnet_dir = download_trashnet_trash()

    # Step 3: Collect all new images
    all_new = []
    for root, dirs, files in os.walk(hf_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                all_new.append(os.path.join(root, f))
    for root, dirs, files in os.walk(trashnet_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                all_new.append(os.path.join(root, f))

    print(f'\nTotal new images collected: {len(all_new)}')

    if len(all_new) == 0:
        print('ERROR: No new images downloaded!')
        return

    # Step 4: Deduplicate
    print('\nDeduplicating new images...')
    dups = deduplicate_by_hash(TEMP_DIR)
    print(f'Removed {dups} duplicates')

    # Re-count after dedup
    all_new = []
    for root, dirs, files in os.walk(TEMP_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                all_new.append(os.path.join(root, f))
    print(f'New images after dedup: {len(all_new)}')

    # Step 5: Merge with existing non-recyclable images
    print('\n' + '=' * 60)
    print('Merging with existing dataset (70/15/15 split)...')
    print('=' * 60)

    # Collect ALL existing non-recyclable images
    existing_nonrec = []
    for split in ['train', 'valid', 'test']:
        dir_path = os.path.join(OUTPUT_DIR, split, 'bukan_daur_ulang')
        if os.path.isdir(dir_path):
            for f in os.listdir(dir_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    existing_nonrec.append(os.path.join(dir_path, f))

    print(f'Existing non-recyclable: {len(existing_nonrec)}')
    print(f'New images: {len(all_new)}')

    # Combine all, shuffle
    all_nonrec = existing_nonrec + all_new
    random.shuffle(all_nonrec)

    print(f'Total non-recyclable images: {len(all_nonrec)}')

    # Clear existing non-recyclable dirs
    for split in ['train', 'valid', 'test']:
        dir_path = os.path.join(OUTPUT_DIR, split, 'bukan_daur_ulang')
        if os.path.isdir(dir_path):
            for f in os.listdir(dir_path):
                fpath = os.path.join(dir_path, f)
                if os.path.isfile(fpath):
                    os.remove(fpath)

    # Split 70/15/15
    n = len(all_nonrec)
    n_train = int(n * 0.7)
    n_valid = int(n * 0.15)

    train_imgs = all_nonrec[:n_train]
    valid_imgs = all_nonrec[n_train:n_train + n_valid]
    test_imgs = all_nonrec[n_train + n_valid:]

    def copy_images(images, split_name, label='bukan_daur_ulang'):
        target = os.path.join(OUTPUT_DIR, split_name, label)
        os.makedirs(target, exist_ok=True)
        count = 0
        for src in images:
            ext = os.path.splitext(src)[1].lower()
            if ext not in ('.jpg', '.jpeg', '.png', '.bmp'):
                ext = '.jpg'
            dst = os.path.join(target, f'nr_{count}{ext}')
            shutil.copy2(src, dst)
            count += 1
        return count

    c_train = copy_images(train_imgs, 'train')
    c_valid = copy_images(valid_imgs, 'valid')
    c_test = copy_images(test_imgs, 'test')

    print(f'\n  train/bukan_daur_ulang: {c_train}')
    print(f'  valid/bukan_daur_ulang: {c_valid}')
    print(f'  test/bukan_daur_ulang : {c_test}')
    print(f'  total                 : {c_train + c_valid + c_test}')

    # Step 6: Cleanup temp
    print('\n' + '=' * 60)
    print('Cleaning up temp files...')
    print('=' * 60)
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        print(f'Removed: {TEMP_DIR}')

    # Final summary
    total_rec_new = 0
    total_nonrec_new = 0
    for split in ['train', 'valid', 'test']:
        d = os.path.join(OUTPUT_DIR, split, 'daur_ulang')
        nd = os.path.join(OUTPUT_DIR, split, 'bukan_daur_ulang')
        dc = len([f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]) if os.path.isdir(d) else 0
        ndc = len([f for f in os.listdir(nd) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]) if os.path.isdir(nd) else 0
        total_rec_new += dc
        total_nonrec_new += ndc
        print(f'  {split}/daur_ulang: {dc} | {split}/bukan_daur_ulang: {ndc}')

    print(f'\n  TOTAL daur_ulang     : {total_rec_new}')
    print(f'  TOTAL bukan_daur_ulang: {total_nonrec_new}')

    if total_rec_new >= 1000 and total_nonrec_new >= 1000:
        print('\n  TARGET TERCAPAI: Kedua kelas >= 1000 gambar!')
    else:
        if total_rec_new < 1000:
            print(f'\n  PERINGATAN: daur_ulang kurang {1000 - total_rec_new} gambar')
        if total_nonrec_new < 1000:
            print(f'\n  PERINGATAN: bukan_daur_ulang kurang {1000 - total_nonrec_new} gambar')

    print(f'\nDataset siap! Jalankan training:')
    print(f'  python train.py')


if __name__ == '__main__':
    main()
