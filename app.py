import os
import io
import math
import base64

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


PRODUCT_NAME = "RETINASCOPE"
MODEL_PATH = "best_densenet121.pth"
NUM_CLASSES = 5
CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]
CLASS_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
CLASS_DESC = [
    "No visible signs of retinopathy.",
    "Microaneurysms only.",
    "More than microaneurysms, less than severe.",
    "Extensive hemorrhages, venous beading, IRMA.",
    "Neovascularization present — highest risk.",
]
SEVERITY_COLORS = ["#3FB88F", "#8FBF56", "#E0B23E", "#E2823F", "#DB4F42"]
AMBER = "#E3A548"
CYAN = "#5FC9D6"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title=f"{PRODUCT_NAME} — Diabetic Retinopathy Screening",
    page_icon="\U0001F441",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo+Black&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
    --bg:#0A0D11;
    --bg-raise:#0F131A;
    --panel:#141920;
    --panel-2:#181F27;
    --border:#262E38;
    --border-soft:#1C232B;
    --text:#E7ECF2;
    --text-dim:#94A0AF;
    --text-faint:#5A6472;
    --amber:#E3A548;
    --amber-soft:rgba(227,165,72,0.14);
    --cyan:#5FC9D6;
    --radius:14px;
    --shadow:0 1px 2px rgba(0,0,0,0.35), 0 10px 30px rgba(0,0,0,0.35);
}

html, body, [class*="css"]{
    font-family:'IBM Plex Sans', sans-serif;
    color:var(--text);
}
.stApp, [data-testid="stAppViewContainer"]{
    background:
        radial-gradient(ellipse 900px 500px at 15% -10%, rgba(227,165,72,0.06), transparent 60%),
        radial-gradient(ellipse 700px 500px at 100% 0%, rgba(95,201,214,0.05), transparent 55%),
        var(--bg);
}
[data-testid="stHeader"]{ background:rgba(0,0,0,0); }
[data-testid="stSidebar"]{
    background:var(--bg-raise);
    border-right:1px solid var(--border-soft);
}
[data-testid="stSidebar"] *{ color:var(--text); }
[data-testid="stSidebar"] hr{ border-color:var(--border-soft); }

.mono{ font-family:'IBM Plex Mono', monospace; }

.topstrip{
    display:flex; align-items:center; justify-content:space-between;
    padding:12px 20px;
    background:var(--panel);
    border:1px solid var(--border-soft);
    border-radius:10px;
    margin-bottom:20px;
}
.topstrip-left{ display:flex; align-items:center; gap:10px; }
.led{
    width:8px; height:8px; border-radius:50%;
    background:#4ADE80;
    box-shadow:0 0 0 3px rgba(74,222,128,0.15), 0 0 8px rgba(74,222,128,0.7);
    animation:pulse 2.4s ease-in-out infinite;
}
@keyframes pulse{
    0%,100%{ opacity:1; }
    50%{ opacity:0.45; }
}
.topstrip-title{
    font-family:'IBM Plex Mono', monospace;
    font-size:11.5px;
    letter-spacing:0.16em;
    color:var(--text-dim);
    text-transform:uppercase;
}
.topstrip-right{
    font-family:'IBM Plex Mono', monospace;
    font-size:11px;
    color:var(--text-faint);
    letter-spacing:0.04em;
    display:flex; gap:18px;
}
.topstrip-right span b{ color:var(--text-dim); font-weight:600; }

.plate{
    border:1px solid var(--border);
    background:linear-gradient(180deg, var(--panel) 0%, var(--bg-raise) 100%);
    border-radius:var(--radius);
    padding:30px 34px;
    margin-bottom:22px;
    box-shadow:var(--shadow);
}
.eyebrow{
    font-family:'IBM Plex Mono', monospace;
    font-size:11px;
    letter-spacing:0.2em;
    color:var(--amber);
    text-transform:uppercase;
    margin-bottom:10px;
}
.plate h1{
    font-family:'Archivo Black', sans-serif;
    font-size:32px;
    font-weight:400;
    letter-spacing:-0.01em;
    margin:0 0 10px 0;
    color:var(--text);
}
.plate p{
    color:var(--text-dim);
    font-size:14.5px;
    max-width:620px;
    line-height:1.55;
    margin:0;
}
.plate-meta{
    font-family:'IBM Plex Mono', monospace;
    font-size:11.5px;
    color:var(--text-faint);
    margin-top:18px;
    border-top:1px dashed var(--border);
    padding-top:12px;
}

.card{
    border:1px solid var(--border);
    background:var(--panel);
    border-radius:var(--radius);
    padding:20px 22px;
    height:100%;
    box-shadow:var(--shadow);
}
.card-label{
    font-family:'IBM Plex Mono', monospace;
    font-size:11px;
    letter-spacing:0.16em;
    text-transform:uppercase;
    color:var(--text-faint);
    margin-bottom:14px;
}

.scope-wrap{ text-align:center; }
.scope{
    position:relative;
    width:100%;
    max-width:280px;
    aspect-ratio:1;
    margin:0 auto 14px auto;
}
.scope svg.ring{ position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
.scope-inner{
    position:absolute;
    inset:15%;
    border-radius:50%;
    overflow:hidden;
    border:2px solid var(--border);
    box-shadow:
        inset 0 0 0 1px rgba(0,0,0,0.4),
        0 0 0 6px rgba(0,0,0,0.25),
        0 8px 24px rgba(0,0,0,0.5);
    background:#000;
}
.scope-inner img{
    width:100%; height:100%;
    object-fit:cover;
    display:block;
    filter:saturate(1.05);
}
.scope-label{
    font-family:'IBM Plex Mono', monospace;
    font-size:11px;
    letter-spacing:0.14em;
    text-transform:uppercase;
    color:var(--text-dim);
    margin-top:2px;
}
.scope-sub{
    font-family:'IBM Plex Mono', monospace;
    font-size:10.5px;
    color:var(--text-faint);
}

.console{
    border:1px solid var(--border);
    background:var(--panel);
    border-radius:var(--radius);
    padding:26px 30px;
    box-shadow:var(--shadow);
}
.console-top{
    display:flex; justify-content:space-between; align-items:flex-start;
    flex-wrap:wrap; gap:20px;
}
.grade-word{
    font-family:'Archivo Black', sans-serif;
    font-weight:400;
    font-size:40px;
    line-height:1.05;
    letter-spacing:-0.01em;
    margin:2px 0 0 0;
}
.digital-readout{
    text-align:right;
}
.digital-label{
    font-family:'IBM Plex Mono', monospace;
    font-size:10.5px;
    letter-spacing:0.16em;
    text-transform:uppercase;
    color:var(--text-faint);
    display:block;
    margin-bottom:4px;
}
.digital-value{
    font-family:'IBM Plex Mono', monospace;
    font-size:32px;
    font-weight:600;
    color:var(--amber);
    text-shadow:0 0 18px rgba(227,165,72,0.35);
}

.dial-wrap{ text-align:center; margin-top:6px; }
.dial-tick-label{ font-family:'IBM Plex Mono', monospace; }

.prob-row{
    display:flex;
    align-items:center;
    gap:14px;
    padding:9px 0;
}
.prob-name{
    width:150px;
    flex-shrink:0;
    font-size:13px;
    color:var(--text-dim);
}
.prob-track{
    flex:1;
    height:8px;
    background:var(--panel-2);
    border:1px solid var(--border-soft);
    border-radius:5px;
    overflow:hidden;
}
.prob-fill{
    height:100%;
    border-radius:5px 0 0 5px;
}
.prob-pct{
    width:54px;
    text-align:right;
    font-family:'IBM Plex Mono', monospace;
    font-size:12.5px;
    color:var(--text-faint);
    flex-shrink:0;
}
.prob-row.active .prob-name{ color:var(--text); font-weight:600; }
.prob-row.active .prob-pct{ color:var(--text); }
.scale-row{
    display:flex;
    gap:10px;
    padding:9px 0;
    border-bottom:1px solid var(--border-soft);
}
.scale-row:last-child{ border-bottom:none; }
.dot{
    width:9px; height:9px; border-radius:50%;
    margin-top:5px; flex-shrink:0;
    box-shadow:0 0 6px currentColor;
}
.scale-name{ font-size:13.5px; font-weight:600; margin-bottom:2px; color:var(--text); }
.scale-desc{ font-size:12px; color:var(--text-faint); line-height:1.45; }

[data-testid="stButton"] button{
    background:var(--amber);
    color:#14100A;
    border:none;
    font-weight:700;
    font-family:'IBM Plex Sans', sans-serif;
    letter-spacing:0.01em;
    border-radius:9px;
    padding:0.6rem 1.5rem;
    box-shadow:0 2px 0 rgba(0,0,0,0.2), 0 0 22px rgba(227,165,72,0.25);
    transition:transform 0.12s ease, box-shadow 0.12s ease;
}
[data-testid="stButton"] button:hover{
    background:#EFAF57;
    color:#14100A;
    transform:translateY(-1px);
    box-shadow:0 3px 0 rgba(0,0,0,0.2), 0 0 26px rgba(227,165,72,0.4);
}

[data-testid="stFileUploaderDropzone"]{
    background:var(--panel-2);
    border:1.5px dashed var(--border);
    border-radius:var(--radius);
}
[data-testid="stFileUploaderDropzone"] *{ color:var(--text-dim) !important; }
[data-testid="stFileUploaderDropzone"] svg{ fill:var(--text-faint) !important; }

.footer-note{
    margin-top:28px;
    padding-top:16px;
    border-top:1px solid var(--border-soft);
    color:var(--text-faint);
    font-size:12px;
    line-height:1.6;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file '{MODEL_PATH}' not found. "
            f"Place it in the same folder as app.py."
        )
        st.stop()

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


with st.spinner("Loading model..."):
    model = load_model()

target_layers = [model.features.denseblock4]
cam = GradCAM(model=model, target_layers=target_layers)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


def predict_and_explain(image: Image.Image):
    image_rgb = image.convert("RGB")
    input_tensor = transform(image_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())

    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    rgb_img = np.array(image_rgb.resize((224, 224))).astype(np.float32) / 255.0
    cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return pred_idx, probs.detach().cpu().numpy(), cam_overlay


def img_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def render_calibration_ring(active_color: str = None) -> str:
    """Static tick-marked ring, like the diopter ring on an ophthalmoscope lens."""
    size, cx, cy, r_out, r_in = 280, 140, 140, 138, 122
    ring_color = active_color or "#333D49"
    parts = [f'<svg class="ring" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">']
    parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="{(r_out + r_in) / 2}" fill="none" '
        f'stroke="{ring_color}" stroke-width="{r_out - r_in}" opacity="0.35"/>'
    )
    for deg in range(0, 360, 6):
        rad = math.radians(deg)
        major = deg % 30 == 0
        r1 = r_in - 2 if major else r_in + 3
        r2 = r_out + 2 if major else r_out - 3
        x1, y1 = cx + r1 * math.cos(rad), cy + r1 * math.sin(rad)
        x2, y2 = cx + r2 * math.cos(rad), cy + r2 * math.sin(rad)
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{"#8892A0" if major else "#3A4350"}" '
            f'stroke-width="{1.4 if major else 0.8}"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def render_scope(image_b64: str, label: str, sub: str, ring_color: str = None) -> str:
    return (
        '<div class="scope-wrap">'
        f'<div class="scope">{render_calibration_ring(ring_color)}'
        f'<div class="scope-inner"><img src="data:image/png;base64,{image_b64}"/></div>'
        '</div>'
        f'<div class="scope-label">{label}</div>'
        f'<div class="scope-sub mono">{sub}</div>'
        '</div>'
    )


def render_dial_gauge(current_idx: int) -> str:
    """Semicircular instrument dial, 5 clinical bands, needle marks the graded stage."""
    w, h = 420, 230
    cx, cy, r_out, r_in = 210, 200, 170, 132
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg">']

    def polar(radius, theta_deg):
        t = math.radians(theta_deg)
        return cx + radius * math.cos(t), cy - radius * math.sin(t)

    band_span = 180 / NUM_CLASSES
    for i, color in enumerate(SEVERITY_COLORS):
        start = 180 - band_span * i
        end = 180 - band_span * (i + 1)
        is_active = (i == current_idx)
        r_mid = (r_out + r_in) / 2
        x1, y1 = polar(r_mid, start)
        x2, y2 = polar(r_mid, end)
        opacity = "1" if is_active else "0.28"
        parts.append(
            f'<path d="M {x1:.1f} {y1:.1f} A {r_mid} {r_mid} 0 0 1 {x2:.1f} {y2:.1f}" '
            f'fill="none" stroke="{color}" stroke-width="{r_out - r_in}" '
            f'stroke-opacity="{opacity}" stroke-linecap="butt"/>'
        )
        lx, ly = polar(r_out + 16, (start + end) / 2)
        weight = "600" if is_active else "400"
        fill = color if is_active else "#5A6472"
        parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" font-size="11.5" '
            f'font-weight="{weight}" class="dial-tick-label" fill="{fill}">{i}</text>'
        )

    needle_angle = 180 - band_span * (current_idx + 0.5)
    nx, ny = polar(r_in - 14, needle_angle)
    parts.append(
        f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" '
        f'stroke="#E7ECF2" stroke-width="2.5" stroke-linecap="round"/>'
    )
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="#E7ECF2"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="3" fill="{SEVERITY_COLORS[current_idx]}"/>')

    parts.append("</svg>")
    return "".join(parts)


    st.markdown('<div class="eyebrow">Reference</div>', unsafe_allow_html=True)
    st.markdown("### Severity scale")
    rows = ""
    for color, label, desc in zip(SEVERITY_COLORS, CLASS_LABELS, CLASS_DESC):
        rows += (
            f'<div class="scale-row">'
            f'<div class="dot" style="background:{color}; color:{color};"></div>'
            f'<div><div class="scale-name">{label}</div>'
            f'<div class="scale-desc">{desc}</div></div></div>'
        )
    st.markdown(rows, unsafe_allow_html=True)

    st.markdown("### Grad-CAM")
    st.markdown(
        '<p style="font-size:12.5px; color:var(--text-faint); line-height:1.6;">'
        "Grad-CAM highlights the retinal regions that most influenced the "
        "prediction. Warmer colors mark areas the model weighted heavily; "
        "cooler areas contributed less.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="footer-note mono">MODEL DenseNet-121<br>'
        f'DEVICE {str(DEVICE).upper()}<br>EXPLAIN GradCAM</div>',
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="topstrip">'
    '<div class="topstrip-left"><div class="led"></div>'
    '<div class="topstrip-title">SYSTEM READY</div></div>'
    '<div class="topstrip-right">'
    f'<span>MODEL <b>DENSENET-121</b></span>'
    f'<span>DEVICE <b>{str(DEVICE).upper()}</b></span>'
    '<span>EXPLAIN <b>GRAD-CAM</b></span>'
    '</div></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="plate">'
    '<div class="eyebrow">Retinal Screening Instrument · 5-Stage Grading</div>'
    f'<h1>{PRODUCT_NAME}</h1>'
    '<p>Upload a fundus photograph to grade diabetic retinopathy severity '
    'and view the Grad-CAM attention map behind the model\'s decision.</p>'
    '<div class="plate-meta">CLASSES 5 &nbsp;·&nbsp; ARCHITECTURE DenseNet-121 '
    '&nbsp;·&nbsp; INPUT 224×224</div>'
    '</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Fundus image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)

if uploaded_file is None:
    st.markdown(
        '<p style="color:var(--text-faint); font-size:13.5px; margin-top:-6px;">'
        "Accepted formats: JPG, JPEG, PNG. The image will be resized to 224×224 "
        "before analysis.</p>",
        unsafe_allow_html=True,
    )

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    run = st.button("Analyze image", type="primary")

    if run:
        with st.spinner("Analyzing..."):
            pred_idx, probs, cam_overlay = predict_and_explain(image)

        label = CLASS_LABELS[pred_idx]
        color = SEVERITY_COLORS[pred_idx]
        confidence = float(probs[pred_idx]) * 100

        orig_b64 = img_to_b64(image)
        cam_b64 = img_to_b64(Image.fromarray(cam_overlay.astype("uint8")))

        st.write("")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                render_scope(orig_b64, "Original", "FUNDUS · 224×224"),
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                render_scope(cam_b64, "Grad-CAM attention", "DENSEBLOCK4 · HEATMAP", color),
                unsafe_allow_html=True,
            )

        st.write("")

        st.markdown(
            '<div class="console">'
            '<div class="console-top">'
            '<div><div class="eyebrow">Predicted stage</div>'
            f'<div class="grade-word" style="color:{color};">{label}</div></div>'
            '<div class="digital-readout">'
            '<span class="digital-label">Confidence</span>'
            f'<span class="digital-value">{confidence:.1f}%</span>'
            '</div>'
            '</div>'
            f'<div class="dial-wrap">{render_dial_gauge(pred_idx)}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Class probabilities</div>', unsafe_allow_html=True)
        bars = ""
        for i, (name, p, color_i) in enumerate(zip(CLASS_LABELS, probs, SEVERITY_COLORS)):
            pct = float(p) * 100
            active = "active" if i == pred_idx else ""
            bars += (
                f'<div class="prob-row {active}">'
                f'<div class="prob-name">{name}</div>'
                f'<div class="prob-track"><div class="prob-fill" '
                f'style="width:{pct:.1f}%; background:{color_i}; '
                f'box-shadow:0 0 10px {color_i}80;"></div></div>'
                f'<div class="prob-pct">{pct:.1f}%</div>'
                f'</div>'
            )
        st.markdown(bars, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="footer-note">This tool is a research prototype and does '
            'not constitute a medical diagnosis. Consult an ophthalmologist for '
            'clinical evaluation.</div>',
            unsafe_allow_html=True,
        )