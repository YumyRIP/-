import streamlit as st
import fitz
from PIL import Image, ImageOps, ImageFilter
import warnings
import numpy as np
warnings.filterwarnings("ignore")

st.set_page_config(page_title="工程圖 PDF OCR 系統", layout="wide")
st.title("🧩 工程圖 PDF OCR 解析系統")
st.caption("上傳 PDF 檔案，轉換成影像後進行 OCR 辨識。")

# ====================== OCR（使用 EasyOCR） ======================
@st.cache_resource
def get_ocr():
    try:
        import easyocr
        return easyocr.Reader(['ch_sim', 'en'], gpu=False)
    except ModuleNotFoundError:
        return None

# ====================== OCR 前處理 ======================
def preprocess_for_ocr(image: Image.Image):
    gray = image.convert("L")
    gray = ImageOps.autocontrast(gray)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    return gray

# ====================== OCR 結果轉文字 ======================
def ocr_results_to_text(results, min_confidence=0.35):
    lines = []
    for item in results:
        if len(item) == 3:
            text = item[1]
            conf = item[2]
        else:
            text = item[1]
            conf = None
        if text and (conf is None or conf >= min_confidence):
            lines.append(text.strip())
    return "\n".join(lines)

ocr = get_ocr()

# ====================== PDF 轉影像 ======================
def pdf_to_image(pdf_bytes: bytes, dpi=300):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return image

# ====================== 影像調整（降低解析度） ======================
def resize_for_ocr(image: Image.Image, max_side=5000):
    width, height = image.size
    if max(width, height) <= max_side:
        return image
    scale = max_side / max(width, height)
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.LANCZOS)

# ====================== 主程式 ======================
uploaded_file = st.file_uploader("上傳 PDF 檔案", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("正在處理 PDF..."):
        try:
            pdf_bytes = uploaded_file.getvalue()
            original_image = pdf_to_image(pdf_bytes, dpi=300)
            st.image(original_image, caption="轉換後的 PDF 第 1 頁影像", width="stretch")

            max_side = st.slider(
                "OCR 最大邊長 (像素)",
                min_value=2000,
                max_value=5000,
                value=4000,
                step=500,
                help="數值越大保留越高解析度，但計算時間也會增加。"
            )

            if ocr is None:
                st.error("OCR 引擎未安裝。請執行 pip install easyocr")
                st.stop()

            with st.spinner("正在執行 OCR..."):
                ocr_image = resize_for_ocr(original_image, max_side=max_side)
                preprocessed = preprocess_for_ocr(ocr_image)
                st.image(preprocessed, caption="OCR 前處理後影像", width=600)

                img_array = np.array(preprocessed)
                ocr_result = ocr.readtext(img_array, detail=1, paragraph=False)
                ocr_text = ocr_results_to_text(ocr_result, min_confidence=0.35)
                if not ocr_text:
                    fallback = ocr.readtext(img_array, detail=0, paragraph=False)
                    ocr_text = "\n".join(fallback) if fallback else ""

            st.subheader("OCR 辨識結果")
            st.text_area("OCR 辯識內容", ocr_text, height=300)
            st.subheader("OCR 詳細項目")
            st.write(ocr_result)

        except Exception as e:
            st.error(f"發生錯誤：{e}")
else:
    st.info("請上傳 PDF 檔案開始測試")

st.caption("Powered by OCR 解析")