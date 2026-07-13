# PaddleOCR VL PDF/PNG 解析範例

## 需求
- Python 3.8+
- `paddleocr`
- `pdf2image`
- `pillow`
- `numpy`
- `opencv-python`

## 安裝

```powershell
python -m pip install -r requirements.txt
```

### Windows 注意
`pdf2image` 需要安裝 Poppler。
1. 下載 Poppler for Windows
2. 解壓後把 `bin` 目錄加入系統 PATH

## 使用方式

```powershell
python paddleocr_vl_pdf_png.py input.pdf
python paddleocr_vl_pdf_png.py input.png
```

結果會輸出到同一個目錄下，檔名為 `input.ocr.txt`。

## 本地網頁介面

```powershell
python app.py
```

或者直接使用整合後的腳本：

```powershell
python paddleocr_vl_pdf_png.py --serve
```

開啟瀏覽器並前往 `http://127.0.0.1:5000`，即可上傳 PDF 或圖片檔案並直接在網頁上檢視 OCR 解析結果。
