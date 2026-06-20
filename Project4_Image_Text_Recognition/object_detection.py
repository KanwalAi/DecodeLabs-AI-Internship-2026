# ============================================================
# Project 4 — Path 2: Object Detection
# Intern   : Kanwal Fatima
# Company  : Decode Labs  |  Batch 2026
# Track    : Artificial Intelligence (AI)
# Backbone : MobileNet v3 + SSD (Single Shot Detector)
# Method   : Transfer Learning — pre-trained on ImageNet + VOC
# Pipeline : Blob Construction → Forward Pass → Softmax → 80% Gate → Draw
# ============================================================

import cv2
import numpy as np
import argparse
import os

# ── CONSTANTS ────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.80   # The Gatekeeper Rule — 80% minimum standard
MODEL_DIR            = "models"
PROTOTXT              = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFEMODEL            = os.path.join(MODEL_DIR, "mobilenet_iter_73000.caffemodel")
OUTPUT_DIR            = "output"
INPUT_SIZE            = 300   # MobileNet-SSD expects 300x300 blobs

# The 21 VOC classes this MobileNet-SSD was trained to recognize
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Distinct colors per class for bounding box visualization
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(VOC_CLASSES), 3), dtype=np.uint8)


def separator(title=""):
    line = "─" * 58
    if title:
        print(f"\n  ┌{line}┐")
        print(f"  │  {title:<56}│")
        print(f"  └{line}┘")


# ── PHASE 1: INPUT — Load Pre-trained Model + Raw Image ───────
def load_model() -> cv2.dnn.Net:
    """
    Transfer Learning: Inheriting the Machine's Knowledge.
    Load MobileNet-SSD — already trained on millions of ImageNet
    images. We attach the pre-built VOC output layer instead of
    training a detector from scratch.
    """
    separator("PHASE 1 · INPUT — Loading Pre-Trained Model (Transfer Learning)")
    if not (os.path.exists(PROTOTXT) and os.path.exists(CAFFEMODEL)):
        raise FileNotFoundError(
            f"Model files not found in '{MODEL_DIR}/'. "
            f"Need: {PROTOTXT} and {CAFFEMODEL}"
        )
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
    print(f"\n  Backbone          : MobileNet v3 (depthwise separable convolutions)")
    print(f"  Detector head     : SSD (Single Shot Detector)")
    print(f"  Trained classes   : {len(VOC_CLASSES)-1} VOC object categories")
    print(f"  Network layers    : {len(net.getLayerNames())}")
    return net


def load_image(path: str) -> np.ndarray:
    """Ingest raw visual data (the IPO Model's INPUT stage)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    h, w, c = img.shape
    print(f"\n  Image loaded     : {path}")
    print(f"  Matrix shape     : {h} (H) × {w} (W) × {c} (Depth/RGB)")
    return img


# ── PHASE 2: PROCESS — Blob Construction + Forward Pass ───────
def build_blob(img: np.ndarray) -> cv2.Mat:
    """
    Step 1: Blob Construction (cv2.dnn.blobFromImage).
      - Scales the image to the required 300×300 network dimensions
      - Performs mean subtraction (centers pixel distribution)
      - Converts BGR → 4D blob: (batch, channels, height, width)
    """
    separator("PHASE 2 · PROCESS — Blob Construction + SSD Forward Pass")
    blob = cv2.dnn.blobFromImage(
        img, scalefactor=0.007843,
        size=(INPUT_SIZE, INPUT_SIZE),
        mean=(127.5, 127.5, 127.5),
        swapRB=False
    )
    print(f"\n  Blob shape        : {blob.shape}  (batch, channels, H, W)")
    print(f"  Resize target     : {INPUT_SIZE}×{INPUT_SIZE} (network input dimensions)")
    print(f"  Mean subtraction  : applied (127.5, 127.5, 127.5)")
    return blob


def run_inference(net: cv2.dnn.Net, blob: cv2.Mat) -> np.ndarray:
    """
    Single Shot Detector forward pass.
    Unlike older 'multiple passes' detectors, SSD classifies AND
    localizes objects in one single network pass — built for
    real-time inference on edge devices.
    """
    net.setInput(blob)
    detections = net.forward()
    print(f"  Forward pass      : complete — {detections.shape[2]} candidate detections generated")
    print(f"  Output format     : normalized (X, Y, W, H) coordinates + Softmax confidence")
    return detections


# ── PHASE 3: OUTPUT — Confidence Gate + Bounding Box Drawing ──
def decode_and_filter(detections: np.ndarray, img: np.ndarray) -> list[dict]:
    """
    Decoding the Machine's Mind: Softmax & Confidence.
    The network doesn't 'know' objects — it outputs a probability
    distribution per class (Softmax). We translate normalized
    coordinates into actual pixel space and apply the 80% Gate:

        if confidence >= 0.80: draw_box_and_label()
        else:                  drop_detection()
    """
    separator("THE 80% GATE — Confidence Filter")

    h, w = img.shape[:2]
    accepted, dropped = [], []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        class_id   = int(detections[0, 0, i, 1])

        if class_id <= 0 or class_id >= len(VOC_CLASSES):
            continue

        if confidence >= CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype(int)
            accepted.append({
                "class": VOC_CLASSES[class_id],
                "class_id": class_id,
                "confidence": confidence,
                "box": (max(0, startX), max(0, startY), min(w, endX), min(h, endY))
            })
        elif confidence > 0.05:   # only log meaningfully-sized rejections, not pure noise
            dropped.append((VOC_CLASSES[class_id] if class_id < len(VOC_CLASSES) else "?", confidence))

    print(f"\n  Threshold required  : {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"  Accepted (PASS)  : {len(accepted)} detections")
    print(f"  ❌ Dropped (FAIL)   : {len(dropped)} below-threshold candidates discarded")
    if dropped:
        for cls, conf in sorted(dropped, key=lambda x: -x[1])[:5]:
            print(f"       · {cls:<12} {conf*100:.1f}%  →  dropped")

    return accepted


def draw_detections(img: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Visual Confirmation: draw accurate bounding boxes with labels."""
    output = img.copy()
    for det in detections:
        (startX, startY, endX, endY) = det["box"]
        color = tuple(int(c) for c in COLORS[det["class_id"]])
        label = f'{det["class"]}: {det["confidence"]*100:.1f}%'

        cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)
        label_y = startY - 10 if startY - 10 > 10 else startY + 20
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output, (startX, label_y - th - 4), (startX + tw + 4, label_y + 4), color, -1)
        cv2.putText(output, label, (startX + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return output


def display_results(detections: list[dict]):
    separator("FINAL RECOGNITION OUTPUT")
    if not detections:
        print("\n  ⚠️  No objects detected above the 80% confidence gate.")
        return
    print(f"\n  ┌{'─'*56}┐")
    print(f"  │  {'OBJECT':<14}{'CONFIDENCE':<14}{'BOUNDING BOX (X,Y,W,H)':<28}│")
    print(f"  ├{'─'*56}┤")
    for det in sorted(detections, key=lambda d: -d["confidence"]):
        x1, y1, x2, y2 = det["box"]
        box_str = f"({x1},{y1},{x2-x1},{y2-y1})"
        print(f"  │  {det['class']:<14}{det['confidence']*100:>6.1f}%{'':<7}{box_str:<28}│")
    print(f"  └{'─'*56}┘")


def save_visual_proof(original_path: str, output_img: np.ndarray):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    basename = os.path.splitext(os.path.basename(original_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{basename}_detected.png")
    cv2.imwrite(out_path, output_img)
    print(f"\n  💾 Visual proof saved: {out_path}")
    return out_path


# ── ENTRY POINT ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Project 4 — Object Detection Pipeline")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--threshold", type=float, default=0.80,
                         help="Confidence threshold (default: 0.80)")
    args = parser.parse_args()

    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.threshold

    print("""
╔════════════════════════════════════════════════════════════╗
║   DecodeLabs Project 4 — Path 2: Object Detection           ║
║   Building the Machine's Optic Nerve                        ║
╚════════════════════════════════════════════════════════════╝""")

    net        = load_model()
    img        = load_image(args.image)
    blob       = build_blob(img)
    detections = run_inference(net, blob)
    accepted   = decode_and_filter(detections, img)
    display_results(accepted)
    output_img = draw_detections(img, accepted)
    save_visual_proof(args.image, output_img)

    separator("DONE")
    print(f"\n  Project 4 (Path 2: Object Detection) complete.\n")


if __name__ == "__main__":
    main()
