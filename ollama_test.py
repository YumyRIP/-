import os
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor
import fitz   

MODEL_PATH = "zai-org/GLM-OCR"
INPUT_PATH = Path(r"C:\Users\Admin\Desktop\新增資料夾\圖檔\12100-5ML-TTMRC3-1 ( 67 x 70 x 23弧 x 79.5 ).pdf")
PDF_PNG_OUTPUT_DIR = Path(r"C:\Users\Admin\Desktop\新增資料夾\PDF轉png存放")

# 建立輸出資料夾
PDF_PNG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_PROMPT = """You are an expert in mechanical engineering drawing interpretation. This is a component engineering drawing that uses a grid reference system with columns A-J (horizontal) and rows 1-8 (vertical). The drawing contains multiple sub-views.

Please analyze the image following these steps:

【Step 1: Identify sub-view regions】
First, list all identifiable sub-views in the drawing along with their approximate grid location, for example:
- Isometric/3D view (with logo)
- Section view A-A
- Detail view B
- Detail view C
- Section view D-D
- Front view
- Side view / cross-section view
- Rear view
- Title block / tolerance table

【Step 2: Analyze each sub-view individually】
For each sub-view listed in Step 1, output the following format separately:

### [Sub-view name]
- Description: (the structure and design features shown in this view)
- Dimensions: (list all numerical dimensions, including units and tolerance symbols, e.g. Ø67 -0.01/-0.02)
- Special symbols/notes: (such as angles, surface treatment symbols, section line direction, etc.)

【Step 3: Title block and tolerance table content】
Please fully and precisely transcribe every field in the title block and tolerance tables, including:
- Company name and drawing title
- Drawing number, material, weight, scale, quantity, revision
- Inner diameter / outer diameter / skirt length / main body length value table
- General machining tolerance table (tolerance values for each dimension range)
- Hole and hole-center distance tolerance table

【Output requirements】
- Use Markdown headers to separate each section — do not merge content from different sub-views
- Preserve all numbers and symbols exactly as shown; do not round or simplify
- If any area is illegible, mark it as "Unreadable" instead of guessing
"""


HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("HF_TOKEN 已設定")
else:
    print("未設定 HF_TOKEN，將以匿名方式嘗試下載模型")

print("Python:", sys.executable)
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("input file exists:", INPUT_PATH.exists())

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"找不到檔案: {INPUT_PATH}")

# 如果是 PDF，先轉換為 PNG
if INPUT_PATH.suffix.lower() == ".pdf":
    print(f"\n檢測到 PDF 檔案: {INPUT_PATH}")
    print("開始轉換 PDF 為 PNG...")
    
    pdf_document = fitz.open(str(INPUT_PATH))
    print(f"PDF 轉換完成，共 {len(pdf_document)} 頁")
    
    # 保存 PNG 到指定資料夾
    saved_png_paths = []
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x 解析度
        
        png_filename = f"{INPUT_PATH.stem}_page{page_num + 1}.png"
        png_path = PDF_PNG_OUTPUT_DIR / png_filename
        pix.save(str(png_path))
        saved_png_paths.append(png_path)
        print(f"已保存: {png_path}")
    
    pdf_document.close()
    
    # 使用第一頁的 PNG 進行分析
    IMAGE_PATH = saved_png_paths[0]
    print(f"\n使用第一頁進行分析: {IMAGE_PATH}\n")
else:
    # 如果是圖片檔案，直接使用
    print("檢測到圖片檔案，直接進行分析\n")
    IMAGE_PATH = INPUT_PATH


print("開始載入 processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("開始載入 model...")
model = AutoModelForImageTextToText.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

print("開始生成辨識結果...")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": str(IMAGE_PATH),
            },
            {
                "type": "text",
                "text": "Text Recognition:",
            },
        ],
    }
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
inputs.pop("token_type_ids", None)

generated_ids = model.generate(**inputs, max_new_tokens=8192)
output_text = processor.decode(
    generated_ids[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=False,
)

output_file = IMAGE_PATH.stem + "_分析結果.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(output_text)

print(f"分析結果已保存到: {output_file}")
