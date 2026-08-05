import cv2
import numpy as np


"""
Fingerprint Quality Assessment Pipeline

Metrics
-------
1. Blur detection
2. Brightness detection
3. Glare detection
4. ROI completeness
5. Ridge clarity

Author: Uday
"""


# ==========================================================
# PREPROCESSING
# ==========================================================

def preprocess_image(image):
    """
    Resize image, convert to grayscale, and apply CLAHE.
    """

    image = cv2.resize(image, (512, 512))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return enhanced


# ==========================================================
# BLUR DETECTION
# ==========================================================

def check_blur(image_bgr):

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return {

        "blur_score": float(blur_score),

        "is_blurry": blur_score < 300
    }

# ==========================================================
# BRIGHTNESS DETECTION
# ==========================================================

def check_brightness(image_bgr):

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    brightness = np.mean(gray)

    return {

        "brightness": float(brightness),

        "too_dark": brightness < 85,

        "too_bright": brightness > 210
    }


# ==========================================================
# GLARE DETECTION
# ==========================================================

def check_glare(image_bgr):

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    bright_pixels = np.sum(gray > 250)

    glare_fraction = bright_pixels / gray.size

    mean_intensity = np.mean(gray)

    has_glare = (
        glare_fraction > 0.005 and
        mean_intensity > 120
    )

    return {

        "has_glare": has_glare,

        "glare_fraction": float(glare_fraction)
    }
# ==========================================================
# ROI EXTRACTION
# ==========================================================

def extract_roi(image_bgr):

    gray = preprocess_image(image_bgr)

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    _, thresh = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    mask = np.zeros_like(gray)

    if len(contours) > 0:

        contour = max(
            contours,
            key=cv2.contourArea
        )

        cv2.drawContours(
            mask,
            [contour],
            -1,
            255,
            thickness=-1
        )

    return mask


# ==========================================================
# ROI COMPLETENESS
# ==========================================================

def check_roi_completeness(image_bgr):

    mask = extract_roi(image_bgr)

    roi_fraction = np.sum(mask > 0) / mask.size

    return {
        "roi_fraction": float(roi_fraction),
        "roi_complete": roi_fraction > 0.15
    }


# ==========================================================
# GABOR FILTER BANK
# ==========================================================

def create_gabor_bank():

    kernels = []

    for theta in np.arange(0, np.pi, np.pi / 8):

        kernel = cv2.getGaborKernel(
            ksize=(21, 21),
            sigma=5,
            theta=theta,
            lambd=10,
            gamma=0.5,
            psi=0,
            ktype=cv2.CV_32F
        )

        kernels.append(kernel)

    return kernels


# ==========================================================
# RIDGE CLARITY
# ==========================================================

def check_ridge_clarity(image_bgr):

    gray = preprocess_image(image_bgr)

    mask = extract_roi(image_bgr)

    kernels = create_gabor_bank()

    responses = []

    for kernel in kernels:

        filtered = cv2.filter2D(
            gray,
            cv2.CV_32F,
            kernel
        )

        responses.append(filtered)

    maximum_response = np.max(
        np.array(responses),
        axis=0
    )

    ridge_pixels = maximum_response[mask > 0]

    if len(ridge_pixels) == 0:
        ridge_score = 0.0
    else:
        ridge_score = np.var(ridge_pixels)

    return {
    "ridge_score": float(ridge_score),
    "ridges_clear": ridge_score > 100000}


# ==========================================================
# COMPOSITE SCORE
# ==========================================================

def compute_score(
        blur,
        brightness,
        glare,
        roi,
        ridge):

    blur_score = min(
        blur["blur_score"] / 50,
        1.0
    )

    brightness_score = 1 - abs(
        brightness["brightness"] - 128
    ) / 128

    glare_score = 1 - min(
        glare["glare_fraction"] / 0.05,
        1.0
    )

    roi_score = min(
        roi["roi_fraction"] / 0.4,
        1.0
    )

    ridge_score = min(
        ridge["ridge_score"] / 40,
        1.0
    )

    final_score = (

        0.25 * blur_score +
        0.15 * brightness_score +
        0.15 * glare_score +
        0.20 * roi_score +
        0.25 * ridge_score

    ) * 100

    final_score = np.clip(
        final_score,
        0,
        100
    )

    return float(final_score)


# ==========================================================
# GUIDANCE SYSTEM
# ==========================================================

def generate_guidance(results):

    if results["brightness"]["too_dark"]:
        return "Increase the lighting."

    if results["glare"]["has_glare"]:
        return "Avoid direct light."

    if results["brightness"]["too_bright"]:
        return "Reduce the light intensity."

    if results["blur"]["is_blurry"]:
        return "Hold the phone steady."

    if not results["roi"]["roi_complete"]:
        return "Move the finger closer."

    if not results["ridge"]["ridges_clear"]:
        return "Fingerprint ridges are unclear."

    return "Good capture — ready for processing."


# ==========================================================
# MASTER QUALITY FUNCTION
# ==========================================================

def quality_gate(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Unable to load image: {image_path}"
        )

    blur = check_blur(image)

    brightness = check_brightness(image)

    glare = check_glare(image)

    roi = check_roi_completeness(image)

    ridge = check_ridge_clarity(image)

    score = compute_score(
        blur,
        brightness,
        glare,
        roi,
        ridge
    )

    passed = (
        score >= 60
        and not blur["is_blurry"]
        and not brightness["too_dark"]
        and not brightness["too_bright"]
        and not glare["has_glare"]
        and roi["roi_complete"]
        and ridge["ridges_clear"]
    )

    results = {

        "passed": passed,

        "composite_score": round(score, 2),

        "blur": blur,

        "brightness": brightness,

        "glare": glare,

        "roi": roi,

        "ridge": ridge
    }

    results["guidance"] = generate_guidance(
        results
    )

    print("\n--------------------------------")
    print("Blur:", blur)
    print("Brightness:", brightness)
    print("Glare:", glare)
    print("ROI:", roi)
    print("Ridge:", ridge)
    print("Score:", round(score, 2))
    print("Passed:", passed)
    print("--------------------------------")

    return results
# ==========================================================
# TESTING
# ==========================================================

if __name__ == "__main__":

    image_path = "sample.jpg"

    try:

        results = quality_gate(image_path)

        print("\nQUALITY ASSESSMENT RESULTS\n")

        for key, value in results.items():
            print(f"{key}: {value}")

    except Exception as e:

        print("Error:", e)