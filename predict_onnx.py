#!/usr/bin/env python3
"""
predict_onnx.py — Prediksi gambar dengan model ONNX Runtime
=============================================================
Memuat model ONNX dan melakukan inferensi menggunakan ONNX Runtime.
Mendukung CPU dan CUDA (jika onnxruntime-gpu terinstall).

Usage:
  python predict_onnx.py --image path/to/image.jpg
  python predict_onnx.py --image path/to/image.jpg --model models/best_model.onnx
  python predict_onnx.py --dir path/to/images/ --provider cuda
"""

import os
import sys
import time
import argparse
import warnings
from typing import List, Dict

import numpy as np
import cv2

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetv2_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess

warnings.filterwarnings('ignore')

CLASS_LABELS = ['daur_ulang', 'bukan_daur_ulang']
IMG_SIZE = 224

PREPROCESS_MAP = {
    'MobileNetV2': mobilenetv2_preprocess,
    'EfficientNetB0': efficientnet_preprocess,
    'ResNet50V2': resnetv2_preprocess,
    'InceptionV3': inceptionv3_preprocess,
    'DenseNet121': densenet_preprocess,
    'NASNetMobile': nasnet_preprocess,
}


def get_onnx_session(model_path: str, provider: str = 'auto'):
    import onnxruntime as ort

    available = ort.get_available_providers()
    print(f'Available providers: {available}')

    if provider == 'auto':
        if 'CUDAExecutionProvider' in available:
            provider = 'CUDAExecutionProvider'
        else:
            provider = 'CPUExecutionProvider'

    if provider not in available:
        print(f'Provider "{provider}" not available, falling back to CPU.')
        provider = 'CPUExecutionProvider'

    print(f'Using provider: {provider}')

    session = ort.InferenceSession(
        model_path,
        providers=[provider],
    )
    return session


def preprocess_image(
    image_path: str, img_size: int = IMG_SIZE, backbone: str = 'EfficientNetB0'
) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f'Cannot read image: {image_path}')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32)
    preprocess_fn = PREPROCESS_MAP[backbone]
    img = preprocess_fn(img)
    img = np.expand_dims(img, axis=0).transpose(0, 3, 1, 2)
    return img


def predict_onnx(
    session, image_path: str, threshold: float = 0.5, backbone: str = 'EfficientNetB0'
) -> Dict:
    img = preprocess_image(image_path, backbone=backbone)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    start = time.perf_counter()
    outputs = session.run([output_name], {input_name: img})
    inference_time = (time.perf_counter() - start) * 1000

    prob = outputs[0][0][0]

    if prob >= threshold:
        label = CLASS_LABELS[0]
        confidence = prob
    else:
        label = CLASS_LABELS[1]
        confidence = 1 - prob

    return {
        'image': os.path.basename(image_path),
        'label': label,
        'confidence': float(confidence),
        'raw_score': float(prob),
        'threshold': threshold,
        'inference_time_ms': round(inference_time, 2),
    }


def predict_batch(
    session, image_dir: str, threshold: float = 0.5, backbone: str = 'EfficientNetB0'
) -> List[Dict]:
    results = []
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    for fname in sorted(os.listdir(image_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in valid_exts:
            continue

        path = os.path.join(image_dir, fname)
        try:
            result = predict_onnx(session, path, threshold, backbone)
            results.append(result)
        except Exception as e:
            print(f'Error processing {fname}: {e}', file=sys.stderr)

    return results


def format_result(result: Dict) -> str:
    icon = '\U0001F504' if result['label'] == 'daur_ulang' else '\u26D4'
    time_str = f'{result["inference_time_ms"]:6.2f}ms' if 'inference_time_ms' in result else ''
    return (
        f'{icon} {result["image"]:30s} '
        f'-> {result["label"]:20s} '
        f'({result["confidence"]:.2%}) {time_str}'
    )


def benchmark(session, image_path: str, num_runs: int = 100, backbone: str = 'EfficientNetB0'):
    img = preprocess_image(image_path, backbone=backbone)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run([output_name], {input_name: img})
        times.append((time.perf_counter() - start) * 1000)

    print(f'\nBenchmark ({num_runs} runs):')
    print(f'  Mean: {np.mean(times):.2f}ms')
    print(f'  Std : {np.std(times):.2f}ms')
    print(f'  Min : {np.min(times):.2f}ms')
    print(f'  Max : {np.max(times):.2f}ms')
    print(f'  FPS : {1000 / np.mean(times):.1f}')


def main():
    parser = argparse.ArgumentParser(
        description='WasteRecycleAI — Predict with ONNX Runtime',
    )
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--dir', type=str, help='Path to directory of images')
    parser.add_argument(
        '--model', type=str, default=None,
        help='Path to .onnx model (auto-selects latest if omitted)',
    )
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument(
        '--provider', type=str, default='auto',
        choices=['auto', 'CPUExecutionProvider', 'CUDAExecutionProvider'],
    )
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark on single image')
    parser.add_argument('--benchmark-runs', type=int, default=100)
    parser.add_argument(
        '--backbone', type=str, default='EfficientNetB0',
        choices=list(PREPROCESS_MAP.keys()),
        help='Backbone used during training (for preprocessing)',
    )

    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.print_help()
        sys.exit(1)

    if args.model:
        model_path = args.model
    else:
        model_dir = 'models'
        onnx_files = sorted([
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir)
            if f.endswith('.onnx')
        ])
        if not onnx_files:
            print('No .onnx model found in models/. Use --model to specify path.')
            sys.exit(1)
        model_path = onnx_files[-1]
        print(f'Auto-selected model: {model_path}')

    if not os.path.exists(model_path):
        print(f'Model not found: {model_path}')
        sys.exit(1)

    print(f'Loading ONNX model: {model_path}')
    session = get_onnx_session(model_path, args.provider)

    input_details = session.get_inputs()[0]
    output_details = session.get_outputs()[0]
    print(f'Input : {input_details.name} {input_details.shape}')
    print(f'Output: {output_details.name} {output_details.shape}')
    print(f'Backbone: {args.backbone}')
    print(f'Threshold: {args.threshold}')
    print('')

    if args.benchmark and args.image:
        benchmark(session, args.image, args.benchmark_runs, args.backbone)
        return

    if args.image:
        if not os.path.exists(args.image):
            print(f'Image not found: {args.image}')
            sys.exit(1)
        result = predict_onnx(session, args.image, args.threshold, args.backbone)
        print(format_result(result))

    if args.dir:
        if not os.path.isdir(args.dir):
            print(f'Directory not found: {args.dir}')
            sys.exit(1)
        results = predict_batch(session, args.dir, args.threshold, args.backbone)
        for r in results:
            print(format_result(r))
        print('')
        labels = [r['label'] for r in results]
        daur = labels.count('daur_ulang')
        bukan = labels.count('bukan_daur_ulang')
        avg_time = np.mean([r['inference_time_ms'] for r in results])
        print(f'Summary: {len(results)} images')
        print(f'  daur_ulang     : {daur}')
        print(f'  bukan_daur_ulang: {bukan}')
        print(f'  Avg inference  : {avg_time:.2f}ms')


if __name__ == '__main__':
    main()
