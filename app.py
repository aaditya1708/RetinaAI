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


PRODUCT_NAME = "RetinaAI"
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
AMBER = "#E3A24B"
TEAL = "#4FB8AE"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title=f"{PRODUCT_NAME} — Diabetic Retinopathy Screening",
    page_icon="\U0001F441",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,500&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root{
    --bg:#0A0E10;
    --bg-raise:#0F1417;
    --surface:#141B1E;
    --surface-2:#182024;
    --border:#232D31;
    --border-soft:#1B2327;
    --ink:#EDF3F1;
    --ink-dim:#96A6A2;
    --ink-faint:#5B6D69;
    --amber:#E3A24B;
    --amber-soft:rgba(227,162,75,0.13);
    --teal:#4FB8AE;
    --teal-soft:rgba(79,184,174,0.13);
    --radius:18px;
    --radius-sm:11px;
    --shadow:0 1px 2px rgba(0,0,0,0.35), 0 16px 36px rgba(0,0,0,0.35);
}

html, body, [class*="css"]{
    font-family:'Inter', sans-serif;
    color:var(--ink);
}
.stApp, [data-testid="stAppViewContainer"]{
    background:
        radial-gradient(ellipse 900px 520px at 10% -8%, rgba(227,162,75,0.05), transparent 60%),
        radial-gradient(ellipse 760px 520px at 100% 0%, rgba(79,184,174,0.045), transparent 55%),
        var(--bg);
}
[data-testid="stHeader"]{ background:rgba(0,0,0,0); }

[data-testid="stSidebar"]{
    background:var(--bg-raise);
    border-right:1px solid var(--border-soft);
}
[data-testid="stSidebar"] *{ color:var(--ink); }
[data-testid="stSidebar"] hr{ border-color:var(--border-soft); margin:22px 0; }
[data-testid="stSidebar"] h3{
    font-family:'Fraunces', serif;
    font-weight:500;
    font-size:19px;
    letter-spacing:-0.005em;
    margin:0 0 14px 0;
    color:var(--ink);
}
[data-testid="stSidebar"] .block-container{ padding-top:2.2rem; }

.mono{ font-family:'JetBrains Mono', monospace; }

/* ---------- status row ---------- */
.status-row{
    display:flex; align-items:center; justify-content:space-between;
    flex-wrap:wrap; gap:10px;
    padding:10px 4px 22px 4px;
    margin-bottom:4px;
    border-bottom:1px solid var(--border-soft);
}
.status-left{ display:flex; align-items:center; gap:9px; }
.led{
    width:7px; height:7px; border-radius:50%;
    background:#57D18A;
    box-shadow:0 0 0 3px rgba(87,209,138,0.14), 0 0 8px rgba(87,209,138,0.6);
    animation:pulse 2.6s ease-in-out infinite;
}
@media (prefers-reduced-motion: reduce){ .led{ animation:none; } }
@keyframes pulse{ 0%,100%{ opacity:1; } 50%{ opacity:0.5; } }
.status-left span{
    font-size:13px;
    color:var(--ink-dim);
}
.status-right{
    display:flex; gap:16px;
    font-family:'JetBrains Mono', monospace;
    font-size:11.5px;
    color:var(--ink-faint);
}
.status-right b{ color:var(--ink-dim); font-weight:500; }

/* ---------- hero ---------- */
.hero{ padding:38px 2px 30px 2px; max-width:640px; }
.hero h1{
    font-family:'Fraunces', serif;
    font-optical-sizing:auto;
    font-weight:500;
    font-size:46px;
    line-height:1.04;
    letter-spacing:-0.015em;
    margin:0 0 14px 0;
    color:var(--ink);
}
.hero p{
    font-size:15.5px;
    line-height:1.6;
    color:var(--ink-dim);
    margin:0;
    max-width:52ch;
}
.hero-stats{
    display:flex; gap:28px; flex-wrap:wrap;
    margin-top:26px;
}
.hero-stat{ display:flex; flex-direction:column; gap:3px; }
.hero-stat .n{
    font-family:'JetBrains Mono', monospace;
    font-size:15px; font-weight:600;
    color:var(--ink);
}
.hero-stat .l{ font-size:11.5px; color:var(--ink-faint); }

/* ---------- upload ---------- */
[data-testid="stFileUploaderDropzone"]{
    background:var(--surface);
    border:1.5px dashed var(--border);
    border-radius:var(--radius);
    transition:border-color 0.15s ease, background 0.15s ease;
}
[data-testid="stFileUploaderDropzone"]:hover{
    border-color:var(--amber);
    background:var(--surface-2);
}
[data-testid="stFileUploaderDropzone"] *{ color:var(--ink-dim) !important; }
[data-testid="stFileUploaderDropzone"] svg{ fill:var(--ink-faint) !important; }
.upload-hint{
    color:var(--ink-faint);
    font-size:13px;
    margin:10px 2px 0 2px;
}

[data-testid="stButton"] button{
    background:var(--amber);
    color:#1A1206;
    border:none;
    font-weight:600;
    font-family:'Inter', sans-serif;
    border-radius:var(--radius-sm);
    padding:0.6rem 1.6rem;
    box-shadow:0 2px 0 rgba(0,0,0,0.2), 0 0 22px rgba(227,162,75,0.22);
    transition:transform 0.12s ease, box-shadow 0.12s ease, background 0.12s ease;
}
[data-testid="stButton"] button:hover{
    background:#EFB05F;
    color:#1A1206;
    transform:translateY(-1px);
    box-shadow:0 3px 0 rgba(0,0,0,0.2), 0 0 26px rgba(227,162,75,0.38);
}
[data-testid="stButton"] button:focus-visible{
    outline:2px solid var(--teal);
    outline-offset:2px;
}

/* ---------- fundus frames ---------- */
.scope-wrap{ text-align:center; padding-top:6px; }
.scope{
    position:relative;
    width:100%;
    max-width:270px;
    aspect-ratio:1;
    margin:0 auto 16px auto;
}
.scope svg.ring{ position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
.scope-inner{
    position:absolute;
    inset:14%;
    border-radius:50%;
    overflow:hidden;
    border:2px solid var(--border);
    box-shadow:
        inset 0 0 0 1px rgba(0,0,0,0.4),
        0 0 0 6px rgba(0,0,0,0.22),
        0 10px 26px rgba(0,0,0,0.5);
    background:#000;
}
.scope-inner img{
    width:100%; height:100%;
    object-fit:cover;
    display:block;
    filter:saturate(1.05);
}
.scope-label{
    font-size:14px;
    font-weight:600;
    color:var(--ink);
    margin-top:2px;
}
.scope-sub{
    font-family:'JetBrains Mono', monospace;
    font-size:11px;
    color:var(--ink-faint);
    margin-top:2px;
}

/* ---------- diagnosis card ---------- */
.console{
    border:1px solid var(--border);
    background:linear-gradient(180deg, var(--surface) 0%, var(--bg-raise) 100%);
    border-radius:var(--radius);
    padding:28px 30px 26px 30px;
    box-shadow:var(--shadow);
}
.console-top{
    display:flex; justify-content:space-between; align-items:flex-end;
    flex-wrap:wrap; gap:18px;
    margin-bottom:22px;
}
.console-label{
    font-size:13px;
    color:var(--ink-faint);
    margin-bottom:6px;
}
.grade-word{
    font-family:'Fraunces', serif;
    font-weight:500;
    font-size:38px;
    line-height:1.05;
    letter-spacing:-0.01em;
    margin:0;
}
.digital-readout{ text-align:right; }
.digital-label{
    font-size:12px;
    color:var(--ink-faint);
    display:block;
    margin-bottom:4px;
}
.digital-value{
    font-family:'JetBrains Mono', monospace;
    font-size:28px;
    font-weight:600;
    color:var(--amber);
}

/* segmented severity track */
.sev-track{ display:flex; gap:6px; }
.sev-seg{
    flex:1;
    height:10px;
    border-radius:5px;
    background:var(--surface-2);
    border:1px solid var(--border-soft);
    position:relative;
    overflow:hidden;
}
.sev-seg .fill{
    position:absolute; inset:0;
    border-radius:5px;
}
.sev-labels{
    display:flex; gap:6px;
    margin-top:9px;
}
.sev-labels span{
    flex:1;
    text-align:center;
    font-size:11px;
    color:var(--ink-faint);
}
.sev-labels span.active{ color:var(--ink); font-weight:600; }

/* ---------- cards ---------- */
.card{
    border:1px solid var(--border);
    background:var(--surface);
    border-radius:var(--radius);
    padding:22px 24px;
    height:100%;
    box-shadow:var(--shadow);
}
.card-label{
    font-size:14px;
    font-weight:600;
    color:var(--ink);
    margin-bottom:16px;
}

.prob-row{ display:flex; align-items:center; gap:14px; padding:9px 0; }
.prob-name{ width:150px; flex-shrink:0; font-size:13.5px; color:var(--ink-dim); }
.prob-track{
    flex:1; height:8px;
    background:var(--surface-2);
    border:1px solid var(--border-soft);
    border-radius:5px;
    overflow:hidden;
}
.prob-fill{ height:100%; border-radius:5px 0 0 5px; }
.prob-pct{
    width:54px; text-align:right;
    font-family:'JetBrains Mono', monospace;
    font-size:12.5px;
    color:var(--ink-faint);
    flex-shrink:0;
}
.prob-row.active .prob-name{ color:var(--ink); font-weight:600; }
.prob-row.active .prob-pct{ color:var(--ink); }

/* ---------- sidebar content ---------- */
.scale-row{ display:flex; gap:10px; padding:9px 0; border-bottom:1px solid var(--border-soft); }
.scale-row:last-child{ border-bottom:none; }
.dot{
    width:9px; height:9px; border-radius:50%;
    margin-top:5px; flex-shrink:0;
    box-shadow:0 0 6px currentColor;
}
.scale-name{ font-size:13.5px; font-weight:600; margin-bottom:2px; color:var(--ink); }
.scale-desc{ font-size:12px; color:var(--ink-faint); line-height:1.45; }
.side-copy{
    font-size:12.5px;
    color:var(--ink-faint);
    line-height:1.6;
}
.side-meta{
    font-family:'JetBrains Mono', monospace;
    font-size:11px;
    color:var(--ink-faint);
    line-height:1.9;
}

.footer-note{
    margin-top:26px;
    padding-top:16px;
    border-top:1px solid var(--border-soft);
    color:var(--ink-faint);
    font-size:12px;
    line-height:1.6;
}

@media (max-width: 640px){
    .hero h1{ font-size:34px; }
    .console-top{ align-items:flex-start; }
    .digital-readout{ text-align:left; }
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
    """Thin instrument ring around the fundus frame, with light cardinal ticks."""
    size, cx, cy, r_out, r_in = 270, 135, 135, 133, 124
    ring_color = active_color or "#333D49"
    parts = [f'<svg class="ring" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">']
    parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="{(r_out + r_in) / 2}" fill="none" '
        f'stroke="{ring_color}" stroke-width="{r_out - r_in}" opacity="0.4"/>'
    )
    for deg in range(0, 360, 45):
        rad = math.radians(deg)
        r1, r2 = r_in - 1, r_out + 3
        x1, y1 = cx + r1 * math.cos(rad), cy + r1 * math.sin(rad)
        x2, y2 = cx + r2 * math.cos(rad), cy + r2 * math.sin(rad)
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#5C6B76" stroke-width="1" opacity="0.55"/>'
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


def render_severity_track(current_idx: int) -> str:
    """Segmented severity track, one segment per clinical stage, active stage lit."""
    segs = ""
    labels = ""
    for i, color in enumerate(SEVERITY_COLORS):
        is_active = (i == current_idx)
        opacity = "1" if is_active else "0.22"
        glow = f"box-shadow:0 0 10px {color}90;" if is_active else ""
        segs += (
            f'<div class="sev-seg">'
            f'<div class="fill" style="background:{color}; opacity:{opacity}; {glow}"></div>'
            f'</div>'
        )
        active_cls = "active" if is_active else ""
        labels += f'<span class="{active_cls}">{i}</span>'
    return f'<div class="sev-track">{segs}</div><div class="sev-labels">{labels}</div>'


with st.sidebar:
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

    st.markdown("---", unsafe_allow_html=True)

    st.markdown("### Grad-CAM")
    st.markdown(
        '<p class="side-copy">Grad-CAM highlights the retinal regions that most '
        'influenced the prediction. Warmer colors mark areas the model weighted '
        'heavily; cooler areas contributed less.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---", unsafe_allow_html=True)

    st.markdown(
        f'<div class="side-meta">Model &nbsp;DenseNet-121<br>'
        f'Device &nbsp;{str(DEVICE).upper()}<br>Explainability &nbsp;Grad-CAM</div>',
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="status-row">'
    '<div class="status-left"><div class="led"></div>'
    '<span>Model loaded and ready</span></div>'
    '<div class="status-right">'
    f'<span>Model <b>DenseNet-121</b></span>'
    f'<span>Device <b>{str(DEVICE).upper()}</b></span>'
    '<span>Explainability <b>Grad-CAM</b></span>'
    '</div></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero">'
    f'<h1>{PRODUCT_NAME}</h1>'
    '<p>Upload a fundus photograph to grade diabetic retinopathy severity '
    'and see the Grad-CAM attention map behind the model\'s decision.</p>'
    '<div class="hero-stats">'
    '<div class="hero-stat"><span class="n">5</span><span class="l">Grading classes</span></div>'
    '<div class="hero-stat"><span class="n">DenseNet-121</span><span class="l">Architecture</span></div>'
    '<div class="hero-stat"><span class="n">224×224</span><span class="l">Input size</span></div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Fundus image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)

if uploaded_file is None:
    st.markdown(
        '<p class="upload-hint">Accepted formats: JPG, JPEG, PNG. The image will '
        'be resized to 224×224 before analysis.</p>',
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
                render_scope(orig_b64, "Original", "Fundus · 224×224"),
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                render_scope(cam_b64, "Grad-CAM attention", "DenseBlock4 · Heatmap", color),
                unsafe_allow_html=True,
            )

        st.write("")

        st.markdown(
            '<div class="console">'
            '<div class="console-top">'
            '<div><div class="console-label">Predicted stage</div>'
            f'<div class="grade-word" style="color:{color};">{label}</div></div>'
            '<div class="digital-readout">'
            '<span class="digital-label">Confidence</span>'
            f'<span class="digital-value">{confidence:.1f}%</span>'
            '</div>'
            '</div>'
            f'{render_severity_track(pred_idx)}'
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