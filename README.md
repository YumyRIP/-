# PDF AI 分析工具

這個程式可以讀取 PDF 檔案，並使用 OpenAI 的 GPT 模型來分析內容。

## 安裝步驟

1. 確保你有 Python 環境。
2. 安裝必要的套件：`pip install streamlit pymupdf pillow numpy easyocr`
3. 設定 OpenAI API 金鑰：
   - 設定環境變數 `OPENAI_API_KEY`，或直接在程式碼中修改。

## 使用方法

1. 將你的 PDF 檔案放在同一個資料夾中。
2. 修改程式中的 `pdf_path` 變數為你的 PDF 檔案名稱。
3. 執行程式：`python 開PDF圖.py`
4. 程式會產生一個 `analysis_result.xlsx` 檔案，包含原始 PDF 內容和 AI 分析結果。

## 注意事項

- 確保 PDF 檔案存在。
- OpenAI API 需要金鑰，請從 OpenAI 官網取得。
- 程式會限制文字長度以避免 API 限制。
