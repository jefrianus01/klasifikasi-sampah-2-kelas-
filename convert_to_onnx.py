#!/usr/bin/env python3
"""
convert_to_onnx.py — Konversi model Keras (.h5) ke ONNX
=========================================================
Mengkonversi model terlatih TensorFlow/Keras ke format ONNX
untuk deployment yang lebih ringan dan cepat.

Usage:
  python convert_to_onnx.py
  python convert_to_onnx.py --model models/my_model.h5
  python convert_to_onnx.py --model models/my_model.h5 --output models/converted.onnx
  python convert_to_onnx.py --opset 15 --quantize
"""

import os
import sys
import argparse
import warnings

import tensorflow as tf
import tf2onnx

warnings.filterwarnings('ignore')


def convert_to_onnx(
    model_path: str,
    output_path: str,
    opset: int = 13,
    quantize: bool = False,
):
    if not os.path.exists(model_path):
        print(f'Error: Model not found: {model_path}')
        sys.exit(1)

    print(f'Loading Keras model: {model_path}')
    model = tf.keras.models.load_model(model_path)
    print(f'Model loaded: {model.name}')
    print(f'Input shape: {model.input_shape}')
    print(f'Output shape: {model.output_shape}')

    dummy = tf.zeros((1, *model.input_shape[1:]), dtype=tf.float32)
    spec = (tf.TensorSpec(dummy.shape, tf.float32, name='input'),)

    model.output_names = ['output']

    print(f'Converting to ONNX (opset {opset})...')
    print(f'Output: {output_path}')

    _, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        output_path=output_path,
        opset=opset,
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'Conversion successful!')
    print(f'ONNX model size: {file_size_mb:.2f} MB')

    if quantize:
        quantize_onnx(output_path)

    return output_path


def quantize_onnx(onnx_path: str):
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quant_path = onnx_path.replace('.onnx', '_quantized.onnx')

        print(f'Quantizing ONNX model (dynamic, uint8)...')
        quantize_dynamic(
            onnx_path,
            quant_path,
            weight_type=QuantType.QUInt8,
        )

        orig_size = os.path.getsize(onnx_path) / (1024 * 1024)
        quant_size = os.path.getsize(quant_path) / (1024 * 1024)
        ratio = (1 - quant_size / orig_size) * 100

        print(f'Quantization complete!')
        print(f'  Original : {orig_size:.2f} MB')
        print(f'  Quantized: {quant_size:.2f} MB ({ratio:.1f}% reduction)')
        print(f'  Saved to : {quant_path}')

    except ImportError:
        print('Warning: onnxruntime-tools not installed. Skipping quantization.')
        print('Install with: pip install onnxruntime-tools')


def main():
    parser = argparse.ArgumentParser(
        description='WasteRecycleAI — Convert Keras Model to ONNX',
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='Path to .h5 model (auto-selects latest if omitted)',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output .onnx path (auto-generated if omitted)',
    )
    parser.add_argument('--opset', type=int, default=13,
                        help='ONNX opset version (default: 13)')
    parser.add_argument('--quantize', action='store_true',
                        help='Apply dynamic quantization after conversion')

    args = parser.parse_args()

    if args.model:
        model_path = args.model
    else:
        model_dir = 'models'
        h5_files = sorted([
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir)
            if f.endswith('.h5')
        ])
        if not h5_files:
            print('No .h5 model found in models/. Use --model to specify path.')
            sys.exit(1)
        model_path = h5_files[-1]
        print(f'Auto-selected model: {model_path}')

    base = os.path.splitext(os.path.basename(model_path))[0]
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(os.path.dirname(model_path) or '.', f'{base}.onnx')

    convert_to_onnx(
        model_path=model_path,
        output_path=output_path,
        opset=args.opset,
        quantize=args.quantize,
    )

    print(f'\nDone! Run prediction with:')
    print(f'  python predict_onnx.py --image path/to/image.jpg --model {output_path}')


if __name__ == '__main__':
    main()
