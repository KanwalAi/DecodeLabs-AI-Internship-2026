# ============================================================
# Project 4 — Path 1: Optical Character Recognition (OCR)
# Intern   : Kanwal Fatima
# Company  : Decode Labs  |  Batch 2026
# Track    : Artificial Intelligence (AI)
# Engine   : pytesseract (Google Tesseract — CNN + Bi-LSTM)
# Pipeline : Grayscale → Gaussian Blur → Deskew → Adaptive Threshold → OCR
# ============================================================

import cv2
import numpy as np
import pytesseract
import argparse
import os

# ── CONSTANTS ────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 80   # The Gatekeeper Rule — 80% minimum standard
OUTPUT_DIR           = "output"


def separator(title=""):
    line = "─" * 58
    if title:
        print(f"\n  ┌{line}┐")
        print(f"  │  {title:<56}│")
        print(f"  └{line}┘")


# ── PHASE 1: INPUT — Load Raw Visual Data ─────────────────────
def load_image(path: str) -> np.ndarray:
    """Ingest raw visual data (the IPO Model's INPUT stage)."""
    separator("PHASE 1 · INPUT — Ingesting Raw Visual Data")
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    h, w, c = img.shape
    print(f"\n  ✅ Image loaded     : {path}")
    print(f"  ✅ Matrix shape     : {h} (H) × {w} (W) × {c} (Depth/RGB)")
    print(f"  ✅ Data points      : {h*w*c:,} individual intensity values (0-255)")
    return img


# ── PHASE 2: PROCESS — The Logic Skeleton (3-step pipeline) ───
def step1_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Step 1: Grayscale Conversion.
    Collapses the 3D RGB matrix into a 1D intensity matrix.
    Removes distracting color data that OCR doesn't need.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"  ✅ Step 1 Grayscale  : 3D RGB → 1D intensity matrix  {img.shape} → {gray.shape}")
    return gray


def step2_denoise(gray: np.ndarray) -> np.ndarray:
    """
    Step 2: Gaussian Blur.
    Smooths the image to eliminate micro-imperfections and artifact noise
    introduced by the scanning/capture process.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print(f"  ✅ Step 2 Denoise    : Gaussian Blur (5×5 kernel) — noise eliminated")
    return blurred


def step3_deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Step 3: Deskewing.
    Calculates the rotation angle of tilted text and snaps it
    back to a perfect horizontal baseline — critical for Tesseract's
    line-based reading order.

    Method: Hough Line Transform on Canny edges. Real scanned text
    forms many near-horizontal baseline/divider edges; the MEDIAN
    angle across all detected lines is a robust skew estimate that
    resists noise far better than a single bounding-box angle.
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                             minLineLength=max(60, gray.shape[1] // 10),
                             maxLineGap=15)

    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:          # keep near-horizontal lines only
                angles.append(angle)

    if len(angles) < 3:
        print(f"  ⚠️  Step 3 Deskew     : insufficient line evidence — skipping rotation")
        return gray, 0.0

    angle = float(np.median(angles))

    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    print(f"  ✅ Step 3 Deskew     : rotation angle detected = {angle:.2f}°  "
          f"(from {len(angles)} baseline edges)  →  corrected to 0°")
    return rotated, angle


def step4_adaptive_threshold(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Adaptive Thresholding (Otsu's Method).
    Forces every pixel to choose a side — pure black or pure white.
        IF pixel_intensity >= cutoff  THEN pixel = 255 (white)
        IF pixel_intensity <  cutoff  THEN pixel = 0   (black)
    Otsu automatically calculates the optimal cutoff value.
    """
    cutoff, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"  ✅ Adaptive Threshold: Otsu cutoff = {cutoff:.0f}  →  pure black-and-white")
    return binary, cutoff


def preprocess_pipeline(img: np.ndarray) -> np.ndarray:
    """Run the full Logic Skeleton: Grayscale → Blur → Deskew → Threshold."""
    separator("PHASE 2 · PROCESS — The Logic Skeleton (Pre-Processing)")
    print()
    gray            = step1_grayscale(img)
    blurred         = step2_denoise(gray)
    deskewed, angle = step3_deskew(blurred)
    binary, cutoff  = step4_adaptive_threshold(deskewed)
    return binary


# ── PHASE 3: OUTPUT — OCR Extraction + Confidence Validation ──
def run_ocr(binary_img: np.ndarray, psm: int = 6) -> tuple[str, float]:
    """
    Run pytesseract on the cleaned binary image.

    PSM (Page Segmentation Mode) tuning, per the Architect's Playbook:
      --psm 3  : Fully automatic (default, varied layouts)
      --psm 6  : Single uniform block of text (documents/invoices)
      --psm 7  : Single text line (plates/headers)
      --psm 11 : Sparse, scattered text

    Returns the extracted text and the mean word-level confidence score.
    """
    separator("PHASE 3 · OUTPUT — OCR Extraction + Confidence Scoring")

    config = f"--psm {psm}"
    text = pytesseract.image_to_string(binary_img, config=config)

    # Get per-word confidence data (Softmax-derived scores from Tesseract's LSTM)
    data = pytesseract.image_to_data(
        binary_img, config=config, output_type=pytesseract.Output.DICT
    )
    confidences = [int(c) for c in data['conf'] if int(c) > 0]
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

    print(f"\n  ✅ PSM mode used     : --psm {psm}")
    print(f"  ✅ Words detected    : {len(confidences)}")
    print(f"  ✅ Mean confidence   : {mean_conf:.1f}%")

    return text.strip(), mean_conf


def apply_confidence_gate(text: str, confidence: float) -> bool:
    """
    The 80% Threshold — The Confidence Filter (Gatekeeper Rule).
    if confidence >= 0.80: accept
    else: flag for review
    """
    separator("THE 80% GATE — Confidence Filter")
    passed = confidence >= CONFIDENCE_THRESHOLD
    gate_symbol = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  Threshold required  : {CONFIDENCE_THRESHOLD}%")
    print(f"  Confidence achieved : {confidence:.1f}%")
    print(f"  Gate result         : {gate_symbol}")
    if not passed:
        print(f"  ⚠️  Below threshold — consider re-scanning or adjusting PSM mode.")
    return passed


def display_results(text: str, confidence: float, passed: bool, original_path: str):
    """Display the final extracted text clearly (Visual Confirmation requirement)."""
    separator("FINAL RECOGNITION OUTPUT")
    print(f"\n  Source image : {original_path}")
    print(f"  Confidence   : {confidence:.1f}%  {'✅' if passed else '⚠️'}")
    print(f"\n  ┌{'─'*56}┐")
    print(f"  │  EXTRACTED TEXT:{' '*39}│")
    print(f"  ├{'─'*56}┤")
    for line in text.split('\n'):
        if line.strip():
            print(f"  │  {line.strip():<54}│")
    print(f"  └{'─'*56}┘")


def save_visual_proof(original_path: str, binary_img: np.ndarray, text: str, confidence: float):
    """Save the pre-processed image as visual confirmation of the pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    basename = os.path.splitext(os.path.basename(original_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{basename}_preprocessed.png")
    cv2.imwrite(out_path, binary_img)
    print(f"\n  💾 Visual proof saved: {out_path}")
    return out_path


# ── ENTRY POINT ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Project 4 — OCR Recognition Pipeline")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--psm", type=int, default=6,
                         help="Tesseract Page Segmentation Mode (default: 6)")
    args = parser.parse_args()

    print("""
╔════════════════════════════════════════════════════════════╗
║   DecodeLabs Project 4 — Path 1: OCR Recognition           ║
║   Building the Machine's Optic Nerve                       ║
╚════════════════════════════════════════════════════════════╝""")

    img             = load_image(args.image)
    binary          = preprocess_pipeline(img)
    text, conf      = run_ocr(binary, psm=args.psm)
    passed          = apply_confidence_gate(text, conf)
    display_results(text, conf, passed, args.image)
    save_visual_proof(args.image, binary, text, conf)

    separator("DONE")
    print(f"\n  ✅ Project 4 (Path 1: OCR) complete.\n")


if __name__ == "__main__":
    main()
