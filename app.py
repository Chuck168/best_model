"""
OP40 Temp-Sensor Check — Streamlit Web App
啟動: streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import pandas as pd
import io, os, csv
from pathlib import Path
from datetime import datetime

# ── 設定 ──────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "best_model_v5.pth"
GITHUB_MANIFEST_URL = "https://raw.githubusercontent.com/Chuck168/best_model/main/model_manifest.json"
LOCAL_VERSION_FILE  = Path(__file__).parent / "local_version.txt"

def fetch_manifest():
    """從 GitHub 取得 manifest，回傳 dict 或 None"""
    import requests
    try:
        return requests.get(GITHUB_MANIFEST_URL, timeout=10).json()
    except Exception:
        return None

def download_model(version_info: dict):
    """下載指定版本的模型，回傳本地 Path"""
    import requests
    filename = version_info["filename"]
    url      = version_info["url"]
    version  = version_info["version"]
    path     = Path(__file__).parent / filename

    local_ver = LOCAL_VERSION_FILE.read_text().strip() if LOCAL_VERSION_FILE.exists() else ""
    if path.exists() and local_ver == version:
        return path  # 已是最新

    try:
        print(f"[Model] Downloading {version} ...")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
        LOCAL_VERSION_FILE.write_text(version)
        print(f"[Model] Done ({len(r.content)/1024/1024:.1f} MB)")
    except Exception as e:
        print(f"[Model] Download failed: {e}")
    return path


# ── 頁面設定 ───────────────────────────────────────────────────
st.set_page_config(
    page_title="OP40 Temp-Sensor Check",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS 樣式 ───────────────────────────────────────────────────
st.markdown("""
<style>
  .ng-badge  { background:#ff3b30; color:white; padding:6px 18px; border-radius:20px;
               font-size:1.2rem; font-weight:700; display:inline-block; }
  .ok-badge  { background:#34c759; color:white; padding:6px 18px; border-radius:20px;
               font-size:1.2rem; font-weight:700; display:inline-block; }
  .conf-text { font-size:0.9rem; color:#888; margin-top:4px; }
  .stat-box  { text-align:center; padding:12px; border-radius:10px;
               background:rgba(0,0,0,0.04); }
  .stat-num  { font-size:2rem; font-weight:800; }
  .stat-lbl  { font-size:0.8rem; color:#888; }
  div[data-testid="stImage"] img { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── v3 顏色感知裁切 ──────────────────────────────────────────
from PIL import ImageFilter

_DARK_THRESH = 155
_WARM_THRESH = 35
_MIN_RATIO   = 0.15
_PADDING     = 1
_W_MIN, _W_MAX = 0.40, 0.60
_H_MIN, _H_MAX = 0.44, 0.58

def color_aware_crop(img):
    arr = np.array(img.convert('RGB'))
    R, B = arr[:,:,0].astype(int), arr[:,:,2].astype(int)
    gray = np.array(img.convert('L'))
    ic_mask = (gray < _DARK_THRESH) | ((R - B) > _WARM_THRESH)
    h, w = gray.shape
    col_ic = ic_mask.mean(axis=0) > _MIN_RATIO
    row_ic = ic_mask.mean(axis=1) > _MIN_RATIO
    if not col_ic.any() or not row_ic.any():
        return img, None, None
    c0 = max(0, np.where(col_ic)[0][0]  - _PADDING)
    c1 = min(w, np.where(col_ic)[0][-1] + _PADDING)
    r0 = max(0, np.where(row_ic)[0][0]  - _PADDING)
    r1 = min(h, np.where(row_ic)[0][-1] + _PADDING)
    wr, hr = (c1-c0)/w, (r1-r0)/h
    return img.crop((c0, r0, c1, r1)), wr, hr



def full_preprocess(img):
    c,wr,hr=color_aware_crop(img.convert('RGB'))
    if wr is None or not (_WMN<=wr<=_WMX and _HMN<=hr<=_HMX): return img
    return c





@st.cache_resource
def load_model(model_path: str = None, version_label: str = "v5"):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    path = Path(model_path) if model_path else MODEL_PATH
    
    # Check which model architecture to use based on the version
    if "v5" in str(version_label).lower() or "resnet" in str(version_label).lower():
        m = models.resnet18(weights=None)
        num_ftrs = m.fc.in_features
        m.fc = nn.Linear(num_ftrs, 2)
    else:
        # V3, V4, V4.1 use MobileNetV2
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.last_channel, 2)
        
    m.load_state_dict(torch.load(str(path), map_location=device, weights_only=True))
    return m.to(device).eval()

# ── predict ───────────────────────────────────────────────────
def predict(model, img: Image.Image, img_size=256):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    local_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    rgb = img.convert("RGB")
    cropped, wr, hr = color_aware_crop(rgb)
    if wr is not None and (_W_MIN<=wr<=_W_MAX and _H_MIN<=hr<=_H_MAX):
        arr = np.array(rgb); R, B = arr[:,:,0].astype(int), arr[:,:,2].astype(int)
        gray = np.array(rgb.convert('L'))
        ic_mask = (gray < _DARK_THRESH) | ((R - B) > _WARM_THRESH)
        h, w = gray.shape
        col_ic = ic_mask.mean(axis=0) > _MIN_RATIO
        row_ic = ic_mask.mean(axis=1) > _MIN_RATIO
        c0 = max(0, np.where(col_ic)[0][0]  - _PADDING)
        c1 = min(w, np.where(col_ic)[0][-1] + _PADDING)
        r0 = max(0, np.where(row_ic)[0][0]  - _PADDING)
        r1 = min(h, np.where(row_ic)[0][-1] + _PADDING)
        bbox = (c0, r0, c1, r1)
        processed = rgb.crop(bbox)
    else:
        bbox = None; processed = rgb
    x = local_tf(processed).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = logits.argmax(1).item()
        # Ensure correct prob mapping. NG is index 0, OK is index 1.
    return pred, probs[0].item(), probs[1].item(), bbox

def result_badge(pred, conf):
    if pred == 0:
        return f'<span class="ng-badge">🔴 NG</span> <span class="conf-text">{conf*100:.1f}%</span>'
    else:
        return f'<span class="ok-badge">🟢 OK</span> <span class="conf-text">{conf*100:.1f}%</span>'

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 OP40 Temp-Sensor Check")
    st.caption("MobileNetV2 分類器")
    st.divider()

    mode = st.radio("模式", ["📷 單張預測", "📁 批次預測"], label_visibility="collapsed")
    st.divider()

    # ── 版本選擇 ──────────────────────────────────────────────
    manifest = fetch_manifest()
    if manifest:
        available = manifest.get("models", [])
        latest_ver = manifest.get("latest", "")
        labels = [m["label"] for m in available]
        # 預設選最新版
        default_idx = next((i for i,m in enumerate(available) if m["version"]==latest_ver), 0)
        selected_label = st.selectbox(
            "🔢 模型版本",
            labels,
            index=default_idx,
            help="預設使用最新版本，也可切換舊版"
        )
        selected_info = next(m for m in available if m["label"]==selected_label)
        MODEL_PATH = download_model(selected_info)
        # 動態更新 img_size（不需要修改 app.py）
        IMG_SIZE = selected_info.get("img_size", 224)
        transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        is_latest = selected_info["version"] == latest_ver
        if is_latest:
            st.caption(f"🟢 使用最新版 **{selected_info['version']}**")
        else:
            st.warning(f"⚠️ 使用舊版 **{selected_info['version']}**")
    else:
        st.warning("⚠️ 無法取得版本清單，使用本機模型")
        selected_info = {"version": "local"}

    st.divider()
    try:
        model = load_model(str(MODEL_PATH), version_label=selected_info.get('version', 'v5'))
        st.success("✅ 模型已就緒")
        st.caption('裝置：' + ('MPS' if torch.backends.mps.is_available() else 'CPU'))
        st.caption("前處理：顏色感知裁切 + 邊緣增強")
        import os
        if MODEL_PATH.exists():
            mtime_str = datetime.fromtimestamp(os.path.getmtime(str(MODEL_PATH))).strftime("%Y-%m-%d %H:%M")
            st.caption(f"最後更新：{mtime_str}")
    except Exception as e:
        st.error(f"模型載入失敗：{e}")
        st.stop()

    st.divider()
    st.caption("OP40 Temp-Sensor Check · 2026-03-27")


# ══════════════════════════════════════════════════════════════
# 模式 1：單張預測
# ══════════════════════════════════════════════════════════════
if mode == "📷 單張預測":
    st.header("📷 單張圖片預測")

    uploaded = st.file_uploader(
        "拖曳或選擇圖片（JPG / PNG）",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        pred, p_ng, p_ok, bbox = predict(model, img)
        conf = p_ng if pred == 0 else p_ok

        # 畫紅框
        from PIL import ImageDraw
        img_display = img.copy()
        if bbox:
            d = ImageDraw.Draw(img_display)
            d.rectangle(bbox, outline=(220, 30, 30), width=4)

        col_img, col_res = st.columns([1, 1], gap="large")

        with col_img:
            st.image(img_display, caption=uploaded.name, use_container_width=True)
            if bbox:
                st.caption("🔴 紅框 = IC 裁切區域")
            else:
                st.warning("⚠️ **未偵測到 IC 裁切區域**\n\n圖片比例或顏色不符合預期，模型以原圖推理，結果可信度較低。", icon="⚠️")

        with col_res:
            st.subheader("判斷結果")
            st.markdown(result_badge(pred, conf), unsafe_allow_html=True)
            st.divider()

            st.write("**信心度分佈**")
            bar_df = pd.DataFrame({
                "類別": ["🔴 NG", "🟢 OK"],
                "機率": [p_ng * 100, p_ok * 100]
            })
            st.bar_chart(bar_df.set_index("類別"), height=200, color=["#ff3b30"])

            st.divider()
            st.markdown(f"""
| 項目 | 數值 |
|------|------|
| 檔案名稱 | `{uploaded.name}` |
| 判斷結果 | {'**🔴 NG**' if pred == 0 else '**🟢 OK**'} |
| NG 機率 | {p_ng*100:.2f}% |
| OK 機率 | {p_ok*100:.2f}% |
""")
    else:
        st.info("👆 請上傳一張感測器圖片開始預測")

# ══════════════════════════════════════════════════════════════
# 模式 2：批次預測
# ══════════════════════════════════════════════════════════════
else:
    st.header("📁 批次圖片預測")

    # 動態 key：每次清除時遞增，讓 file uploader 完全重新渲染
    if "upload_counter" not in st.session_state:
        st.session_state.upload_counter = 0

    col_clear, _ = st.columns([1, 4])
    with col_clear:
        if st.button("🗑 清除所有檔案", use_container_width=True):
            st.session_state.upload_counter += 1
            st.rerun()

    uploaded_files = st.file_uploader(
        "一次上傳多張圖片",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"batch_uploader_{st.session_state.upload_counter}",
    )

    if uploaded_files:
        # ── 進度條預測 ───────────────────────────────────────
        results = []
        progress = st.progress(0, text="準備中…")

        for i, f in enumerate(uploaded_files):
            progress.progress((i + 1) / len(uploaded_files),
                              text=f"處理中 {i+1}/{len(uploaded_files)}：{f.name}")
            try:
                img  = Image.open(f).convert("RGB")
                pred, p_ng, p_ok, bbox = predict(model, img)
                conf = p_ng if pred == 0 else p_ok
                results.append({
                    "檔案名稱": f.name,
                    "結果": "NG" if pred == 0 else "OK",
                    "信心度": f"{conf*100:.1f}%",
                    "NG 機率": f"{p_ng*100:.2f}%",
                    "OK 機率": f"{p_ok*100:.2f}%",
                    "_pred": pred,
                    "_conf": conf,
                    "_img": img,
                    "_bbox": bbox,
                })
            except Exception as e:
                results.append({
                    "檔案名稱": f.name,
                    "結果": "ERROR",
                    "信心度": "-",
                    "NG 機率": "-",
                    "OK 機率": str(e),
                    "_pred": -1, "_conf": 0, "_img": None,
                })

        progress.empty()

        # ── 統計摘要 ─────────────────────────────────────────
        ng_count = sum(1 for r in results if r["_pred"] == 0)
        ok_count = sum(1 for r in results if r["_pred"] == 1)
        total    = len(results)
        ng_rate  = ng_count / total * 100 if total else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{total}</div><div class="stat-lbl">總計</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-box"><div class="stat-num" style="color:#34c759">{ok_count}</div><div class="stat-lbl">OK</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-box"><div class="stat-num" style="color:#ff3b30">{ng_count}</div><div class="stat-lbl">NG</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="stat-box"><div class="stat-num" style="color:#ff9500">{ng_rate:.1f}%</div><div class="stat-lbl">NG 率</div></div>', unsafe_allow_html=True)

        st.divider()

        # ── 篩選器 ───────────────────────────────────────────
        filter_col, _, export_col = st.columns([2, 3, 2])
        with filter_col:
            show = st.selectbox("顯示", ["全部", "僅 NG", "僅 OK"], label_visibility="collapsed")
        with export_col:
            # CSV 匯出
            csv_buf = io.StringIO()
            writer  = csv.DictWriter(csv_buf, fieldnames=["檔案名稱","結果","信心度","NG 機率","OK 機率"])
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in ["檔案名稱","結果","信心度","NG 機率","OK 機率"]})
            st.download_button(
                "⬇️ 匯出 CSV",
                data=csv_buf.getvalue().encode("utf-8-sig"),
                file_name=f"sensor_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # ── 圖片網格結果 ─────────────────────────────────────
        filtered = [r for r in results if
                    show == "全部" or
                    (show == "僅 NG" and r["_pred"] == 0) or
                    (show == "僅 OK" and r["_pred"] == 1)]

        COLS = 4
        for row_start in range(0, len(filtered), COLS):
            cols = st.columns(COLS, gap="small")
            for col_idx, r in enumerate(filtered[row_start:row_start + COLS]):
                with cols[col_idx]:
                    if r["_img"]:
                        img_show = r["_img"].copy()
                        if r.get("_bbox"):
                            from PIL import ImageDraw
                            ImageDraw.Draw(img_show).rectangle(
                                r["_bbox"], outline=(220, 30, 30), width=3)
                        st.image(img_show, use_container_width=True)
                    badge_html = result_badge(r["_pred"], r["_conf"])
                    st.markdown(f"**{r['檔案名稱'][:22]}**", unsafe_allow_html=False)
                    st.markdown(badge_html, unsafe_allow_html=True)

    else:
        st.info("👆 請一次上傳多張圖片進行批次預測")

        # 空白狀態示意
        st.markdown("---")
        st.markdown("""
**使用方式：**
1. 點擊上方區域 或 直接拖曳圖片
2. 支援一次多選（Cmd+A 全選）
3. 自動顯示 NG/OK 結果與信心度
4. 點擊「匯出 CSV」儲存報告
""")
