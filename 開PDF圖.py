import os
os.environ["FLAGS_enable_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["PADDLE_PIR_ENABLE_NEW_IR"] = "0"

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import fitz
from PIL import Image
import io
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")

st.set_page_config(page_title="工程圖逆時鐘解析系統", layout="wide")
st.title("🧩 工程圖逆時鐘解析系統")
st.caption("上傳 PDF → 自動產生 圖A~圖G（精簡穩定版）")

# ====================== API Key ======================
load_dotenv("API.env", override=True)
api_key = os.getenv("XAI_API_KEY")

st.sidebar.header("🔑 API Key 狀態")
st.sidebar.success("✅ API Key 載入成功")

model_name = os.getenv("XAI_MODEL", "grok-2")
st.sidebar.header("🤖 模型設定")
st.sidebar.write(f"使用模型：{model_name}")
st.sidebar.info("如收到 Model not found，請於 API.env 設定 XAI_MODEL")

# ====================== LLM ======================
client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

# ====================== OCR（最精簡） ======================
@st.cache_resource
def get_ocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(lang="ch")   # 只保留 lang

ocr = get_ocr()

# ====================== PDF 轉影像（降低解析度） ======================
def pdf_to_image(pdf_bytes, dpi=260):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=matrix)
    doc.close()
    return Image.open(io.BytesIO(pix.tobytes("png")))

def resize_for_ocr(image: Image.Image, max_side=2000):
    width, height = image.size
    if max(width, height) <= max_side:
        return image
    scale = max_side / max(width, height)
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.LANCZOS)

# ====================== 輔助函式 ======================
def call_llm(prompt: str):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4096
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_text = str(e)
        st.error(f"LLM 呼叫失敗: {error_text}")
        if "Model not found" in error_text or "invalid argument" in error_text:
            st.error(
                "請確認 API.env 中的 XAI_MODEL 是否為可用模型，" \
                "或改用其他可用模型名稱。"
            )
        return None

# ====================== Prompt（你的原始版本） ======================
# ====================== Prompt（完整保留） ======================
SUB_PROMPT = """你是一位專業的機械工程圖尺寸提取專家。

輸入：一張工程圖的完整 OCR 文字。

任務：精準提取所有尺寸、符號與註記，整理成以下 JSON 格式（只輸出 JSON，不要任何其他文字）：

{
  "raw_dimensions": ["內徑 Ø67", "外徑 Ø70", ...],
  "classified": {
    "內徑": "Ø67",
    "外徑": "Ø70",
    "裙長": "23弧",
    "本體長": "79.5",
    "圓角": ["R0.3（2處）", "R0.5"],
    "倒角": "15°",
    "孔中心距": ["32/32/30/30", "33.5/42/31/33.5"],
    "貫穿孔": ["6.8 THRU", "9.2 THRU（2處）"],
    "沉頭孔": ["10H7（2處）"],
    "其他": ["9.2-4", "47.8", "43.5", "※加工配合治具尺寸"]
  }
}"""

MAIN_PROMPT = """你是一位專業的機械工程圖解析專家。

任務：將工程圖依照逆時鐘方向切割成7個圖塊，並嚴格輸出以下 JSON（只輸出 JSON，不要任何其他文字）。

{
  "blocks": [
    {
      "block_id": "圖A",
      "view_name": "左上角側視圖 (Side View)",
      "dimensions": ["內徑 Ø67", "外徑 Ø70", "裙長 23弧", "本體長 79.5", "圓角 R0.3（2處）", "倒角 15°", "圓角 R0.5", "※加工配合治具尺寸"]
    },
    {
      "block_id": "圖B",
      "view_name": "頂部中央角度與圓角細部圖",
      "dimensions": ["倒角 15°", "圓角 R0.3（細部B，2處）", "圓角 R0.5（細部C）"]
    },
    {
      "block_id": "圖C",
      "view_name": "右上角等角立體圖 (Isometric View)",
      "dimensions": ["整體外觀（含散熱鰭片與安裝凸耳）", "Logo: TTMRC"]
    },
    {
      "block_id": "圖D",
      "view_name": "右側正視圖1 (Front View 1)",
      "dimensions": ["外徑 68", "孔中心距 32/32/30/30", "孔中心距 33.5/42/31/33.5", "貫穿孔 6.8 THRU", "貫穿孔 9.2 THRU（2處）", "沉頭孔 10H7（2處）", "其他 9.2-4、47.8、43.5"]
    },
    {
      "block_id": "圖E",
      "view_name": "右下角正視圖2 (Front View 2)",
      "dimensions": ["外徑 68", "孔中心距 32/32/30/30", "孔中心距 33.5/42/31", "貫穿孔 9.2-4（4處）", "沉頭孔 10H7", "其他 9.3-4、9.5-2"]
    },
    {
      "block_id": "圖F",
      "view_name": "中下剖面圖 A-A (Section A-A)",
      "dimensions": ["鰭片相關 27.4/20/41.1/16.5", "鰭片厚度/間距 6.3/6.2", "本體長度方向 79.5", "其他 33.5/42/31/47.8/6.8 THRU"]
    },
    {
      "block_id": "圖G",
      "view_name": "左下角剖面圖 D-D 與主體俯視 (Section D-D & Top View)",
      "dimensions": ["外徑 68", "孔中心距 32/32/30/30", "孔中心距 33.5/42/31", "貫穿孔 6.8 THRU / 9.2 THRU（多處）", "沉頭孔 10H7（2處）", "其他 59.3/27/6.2/43.5/9.2-2"]
    }
  ],
  "common_dimensions": {
    "內徑": "Ø67",
    "外徑": "Ø70",
    "裙長": "23弧",
    "本體長": "79.5"
  }
}

請直接輸出 JSON，不要任何解釋文字。
"""
# ====================== 主程式 ======================
uploaded_file = st.file_uploader("上傳工程圖 PDF（單頁）", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("正在處理 PDF..."):
        try:
            original_image = pdf_to_image(uploaded_file.getvalue(), dpi=260)
            st.image(original_image, caption="原始工程圖 (260 DPI)", width="stretch")

            with st.spinner("正在執行 OCR..."):
                ocr_image = resize_for_ocr(original_image, max_side=2000)
                img_array = np.array(ocr_image)
                result = ocr.ocr(img_array)
                
                ocr_text = "\n".join([line[1][0] for line in result[0] if line]) if result and result[0] else ""
                st.text_area("OCR 辨識結果", ocr_text, height=200)

            with st.spinner("AI 正在進行逆時鐘解析..."):
                sub_prompt_with_ocr = SUB_PROMPT + "\n\nOCR 辨識結果：\n" + ocr_text
                sub_result = call_llm(sub_prompt_with_ocr)
                if not sub_result:
                    st.stop()
                
                final_prompt = MAIN_PROMPT + "\n\n已提取的尺寸參考：\n" + sub_result
                final_json_str = call_llm(final_prompt)
                
                if not final_json_str:
                    st.stop()
                
                start = final_json_str.find("{")
                end = final_json_str.rfind("}") + 1
                clean_json = final_json_str[start:end]
                result_data = json.loads(clean_json)

                st.success("✅ 解析完成！")

                for block in result_data.get("blocks", []):
                    st.subheader(f"{block['block_id']} - {block['view_name']}")
                    st.write("、".join(block.get("dimensions", [])))
                    st.divider()

                st.subheader("共同基準尺寸")
                for k, v in result_data.get("common_dimensions", {}).items():
                    st.write(f"**{k}**：{v}")

        except Exception as e:
            st.error(f"發生錯誤：{e}")
else:
    st.info("請上傳 PDF 檔案開始測試")

st.caption("Powered by Grok | 精簡低負荷版")