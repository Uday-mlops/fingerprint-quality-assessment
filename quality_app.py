import streamlit as st
import tempfile
from quality_assessment import quality_gate

# ==========================================================
# DEFAULT THRESHOLDS
# ==========================================================

BLUR_THRESHOLD = 300
DARK_THRESHOLD = 85
GLARE_THRESHOLD = 0.005
RIDGE_THRESHOLD = 100000
ROI_THRESHOLD = 0.15

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Fingerprint Quality Assessment",
    layout="wide"
)

st.title("🔍 Fingerprint Quality Assessment")

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("Threshold Settings")

blur_threshold = st.sidebar.slider(
    "Blur Threshold",
    0,
    1000,
    BLUR_THRESHOLD
)

dark_threshold = st.sidebar.slider(
    "Dark Threshold",
    0,
    255,
    DARK_THRESHOLD
)

glare_threshold = st.sidebar.slider(
    "Glare Threshold",
    0.0,
    0.05,
    float(GLARE_THRESHOLD)
)

roi_threshold = st.sidebar.slider(
    "ROI Threshold",
    0.0,
    1.0,
    float(ROI_THRESHOLD)
)

ridge_threshold = st.sidebar.slider(
    "Ridge Threshold",
    0,
    500000,
    RIDGE_THRESHOLD
)

st.sidebar.markdown("---")

st.sidebar.subheader("Current Thresholds")

st.sidebar.write("Blur:", blur_threshold)
st.sidebar.write("Dark:", dark_threshold)
st.sidebar.write("Glare:", glare_threshold)
st.sidebar.write("ROI:", roi_threshold)
st.sidebar.write("Ridge:", ridge_threshold)

# ==========================================================
# FILE UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload fingerprint image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================================
# IMAGE PROCESSING
# ==========================================================

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        width=400
    )

    with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
    ) as temp_file:

        temp_file.write(uploaded_file.read())
        image_path = temp_file.name

    results = quality_gate(image_path)

    st.markdown("---")

    col1, col2 = st.columns(2)

    # ======================================================
    # LEFT COLUMN
    # ======================================================

    with col1:

        blur_status = (
            "❌ FAIL"
            if results["blur"]["is_blurry"]
            else "✅ PASS"
        )

        brightness_status = (
            "❌ FAIL"
            if results["brightness"]["too_dark"]
            or results["brightness"]["too_bright"]
            else "✅ PASS"
        )

        glare_status = (
            "❌ FAIL"
            if results["glare"]["has_glare"]
            else "✅ PASS"
        )

        st.metric(
            "Blur Score",
            round(results["blur"]["blur_score"], 2)
        )

        st.write(blur_status)

        st.metric(
            "Brightness",
            round(results["brightness"]["brightness"], 2)
        )

        st.write(brightness_status)

        st.metric(
            "Glare Fraction",
            round(results["glare"]["glare_fraction"], 4)
        )

        st.write(glare_status)

    # ======================================================
    # RIGHT COLUMN
    # ======================================================

    with col2:

        roi_status = (
            "✅ PASS"
            if results["roi"]["roi_complete"]
            else "❌ FAIL"
        )

        ridge_status = (
            "✅ PASS"
            if results["ridge"]["ridges_clear"]
            else "❌ FAIL"
        )

        st.metric(
            "ROI Fraction",
            round(results["roi"]["roi_fraction"], 4)
        )

        st.write(roi_status)

        st.metric(
            "Ridge Score",
            round(results["ridge"]["ridge_score"], 2)
        )

        st.write(ridge_status)

    # ======================================================
    # COMPOSITE SCORE
    # ======================================================

    st.markdown("---")

    score = results["composite_score"]

    if score >= 60:
        st.success(f"Composite Score: {score}")
    else:
        st.error(f"Composite Score: {score}")

    # ======================================================
    # GUIDANCE
    # ======================================================

    st.info(
        f"Guidance: {results['guidance']}"
    )

    # ======================================================
    # FINAL DECISION
    # ======================================================

    st.markdown("---")

    st.subheader("Final Decision")

    if results["passed"]:
        st.success("✅ Fingerprint accepted")
    else:
        st.error("❌ Fingerprint rejected")