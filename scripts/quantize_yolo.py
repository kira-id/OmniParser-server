#!/usr/bin/env python3
"""Export the YOLO icon detector to ONNX and run dynamic quantization."""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit("Please install ultralytics (matching requirements.txt) before running this script") from exc

try:
    from onnxruntime.quantization import QuantType, quantize_dynamic
except ImportError as exc:
    raise SystemExit("Please install onnxruntime to run YOLO quantization (pip install onnxruntime)") from exc

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an ultralytics YOLO checkpoint to ONNX and create a dynamically quantized version",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/icon_detect/model.pt"),
        help="Path to the YOLO weights to export",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("weights/icon_detect_quant"),
        help="Directory where the ONNX and quantized models will be stored",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image width/height used during export",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset to use when exporting",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Run ONNX simplification after export (requires onnxsim)",
    )
    parser.add_argument(
        "--dynamic-shapes",
        action="store_true",
        help="Enable dynamic shapes inside the exported ONNX",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Base name for the exported files (defaults to the weights stem)",
    )
    parser.add_argument(
        "--onnx-file",
        type=Path,
        default=None,
        help="If provided, skip the export step and quantize this existing ONNX file",
    )
    parser.add_argument(
        "--skip-quant",
        action="store_true",
        help="Only export ONNX without running quantization",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    return parser.parse_args()


def export_to_onnx(
    weights: Path,
    output_dir: Path,
    model_name: str,
    imgsz: int,
    opset: int,
    dynamic_shapes: bool,
    simplify: bool,
) -> Path:
    logger.info("Exporting YOLO model %s to ONNX", weights)
    model = YOLO(str(weights))
    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic_shapes,
        simplify=False,
    )
    onnx_path = weights.parent / f"{model_name}.onnx"
    if not onnx_path.exists():
        raise RuntimeError("YOLO export did not produce an ONNX file")
    destination = output_dir / f"{model_name}.onnx"
    shutil.copy2(onnx_path, destination)
    logger.info("Saved ONNX model to %s", destination)
    return destination


def quantize_model(onnx_path: Path, output_dir: Path, model_name: str, overwrite: bool) -> Path:
    quant_path = output_dir / f"{model_name}.quantized.onnx"
    if quant_path.exists() and not overwrite:
        raise FileExistsError(
            "Quantized model already exists. Use --overwrite to regenerate it."
        )
    logger.info("Running ONNX Runtime dynamic quantization")
    quantize_dynamic(str(onnx_path), str(quant_path), weight_type=QuantType.QInt8)
    logger.info("Quantized model saved to %s", quant_path)
    return quant_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    if args.onnx_file is None and not weights.exists():
        raise FileNotFoundError(f"Requested weights file not found: {weights}")

    model_name = args.name or weights.stem
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.onnx_file:
        onnx_path = args.onnx_file.expanduser().resolve()
        if not onnx_path.exists():
            raise FileNotFoundError(f"Provided ONNX file does not exist: {onnx_path}")
        logger.info("Using provided ONNX file %s", onnx_path)
    else:
        onnx_path = export_to_onnx(
            weights=weights,
            output_dir=output_dir,
            model_name=model_name,
            imgsz=args.imgsz,
            opset=args.opset,
            dynamic_shapes=args.dynamic_shapes,
            simplify=args.simplify,
        )

    if args.skip_quant:
        logger.info("Skipping quantization as requested (--skip-quant)")
        return

    quantize_model(onnx_path=onnx_path, output_dir=output_dir, model_name=model_name, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
