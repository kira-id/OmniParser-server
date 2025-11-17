"""FastAPI server exposing the OmniParser inference pipeline."""

from __future__ import annotations

from typing import Optional, Tuple

import argparse
import base64
import binascii
import io
import os
import time

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
import uvicorn

from util.utils import (
    check_ocr_box,
    get_caption_model_processor,
    get_som_labeled_img,
    get_yolo_model,
)

# Initialise models once so the API can serve requests immediately.
yolo_model = get_yolo_model(model_path="weights/icon_detect/model.pt")
caption_model_processor = get_caption_model_processor(
    model_name="florence2", model_name_or_path="weights/icon_caption_florence"
)


def get_models():
    """Return globally cached models (lazy loading hook)."""

    return yolo_model, caption_model_processor


MAX_SIZE = 1600

OMNIPARSER_API_KEY = os.environ.get("OMNIPARSER_API_KEY", "").strip()
if not OMNIPARSER_API_KEY:
    raise RuntimeError("OMNIPARSER_API_KEY environment variable must be provided to start OmniParser")


def _validate_api_key(authorization: str | None = Header(None)) -> None:
    """Ensure requests provide the configured Bearer token."""

    if authorization is None:
        raise HTTPException(status_code=401, detail="Authorization header is required")

    try:
        scheme, token = authorization.split(" ", 1)
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Authorization header must be in the format 'Bearer <token>'",
        )

    if scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Authorization header must use the Bearer scheme")

    if token.strip() != OMNIPARSER_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


def _decode_base64_image(image_base64: str) -> Image.Image:
    """Decode a base64 string into a RGB PIL image."""

    try:
        image_bytes = base64.b64decode(image_base64)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="image_base64 is not valid base64") from exc

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # Pillow can raise multiple error types
        raise HTTPException(status_code=400, detail="image_base64 could not be decoded into an image") from exc

    return image


@torch.inference_mode()
def process(
    image_input: Image.Image,
    box_threshold: float,
    iou_threshold: float,
    use_paddleocr: bool,
    imgsz: int,
) -> Tuple[str, list, float]:
    """Run OmniParser on the provided PIL image."""

    start_time = time.time()
    yolo_model, caption_model_processor = get_models()

    # Resize image for faster processing if too large
    if image_input.size[0] > MAX_SIZE or image_input.size[1] > MAX_SIZE:
        ratio = min(MAX_SIZE / image_input.size[0], MAX_SIZE / image_input.size[1])
        new_size = (int(image_input.size[0] * ratio), int(image_input.size[1] * ratio))
        image_input = image_input.resize(new_size, Image.Resampling.LANCZOS)

    box_overlay_ratio = image_input.size[0] / 3200
    draw_bbox_config = {
        "text_scale": 0.8 * box_overlay_ratio,
        "text_thickness": max(int(2 * box_overlay_ratio), 1),
        "text_padding": max(int(3 * box_overlay_ratio), 1),
        "thickness": max(int(3 * box_overlay_ratio), 1),
    }

    (text, ocr_bbox), _ = check_ocr_box(
        image_input,
        display_img=False,
        output_bb_format="xyxy",
        goal_filtering=None,
        easyocr_args={"paragraph": False, "text_threshold": 0.7},
        use_paddleocr=use_paddleocr,
    )
    dino_labled_img, _, parsed_content_list = get_som_labeled_img(
        image_input,
        yolo_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
        batch_size=64,
    )

    latency = time.time() - start_time
    print(f"Processing completed in {latency:.2f} seconds")
    return dino_labled_img, parsed_content_list, latency


class InferenceRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded screenshot")
    box_threshold: float = Field(0.03, ge=0.01, le=1.0)
    iou_threshold: float = Field(0.9, ge=0.01, le=1.0)
    use_paddleocr: bool = Field(False)
    imgsz: int = Field(448, ge=320, le=1280)
    return_som_image: bool = Field(
        False,
        description="Set true to include the SOM overlay image as base64 in the response",
    )


class InferenceResponse(BaseModel):
    som_image_base64: Optional[str] = None
    parsed_content_list: list
    latency: float


app = FastAPI(title="OmniParser API", description="OmniParser inference service")


@app.get("/probe", summary="Readiness probe")
async def probe():
    return {"status": "ready"}


@app.post("/infer", response_model=InferenceResponse, summary="Run OmniParser inference")
async def infer(request: InferenceRequest, _api_key: None = Depends(_validate_api_key)) -> InferenceResponse:
    image = _decode_base64_image(request.image_base64)
    som_image_base64, parsed_content_list, latency = process(
        image,
        box_threshold=request.box_threshold,
        iou_threshold=request.iou_threshold,
        use_paddleocr=request.use_paddleocr,
        imgsz=request.imgsz,
    )

    return InferenceResponse(
        som_image_base64=som_image_base64 if request.return_som_image else None,
        parsed_content_list=parsed_content_list,
        latency=latency,
    )


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run the OmniParser FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable autoreload (useful during development)",
    )
    return parser.parse_args()


def run_server(host: str, port: int, reload: bool = False):
    """Launch the FastAPI server with uvicorn."""

    uvicorn.run("server:app", host=host, port=port, reload=reload)


def main():
    args = parse_cli_args()
    run_server(args.host, args.port, args.reload)


if __name__ == "__main__":
    main()
