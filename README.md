# GLM-OCR 影像文字辨識執行報告

本報告說明如何使用 `zai-org/GLM-OCR` 模型進行影像文字辨識（OCR）的程式邏輯與執行流程。

---

## 1. 程式碼功能說明

此程式的主要功能是利用 GLM-OCR 視覺語言模型，針對指定的本地圖片進行文字偵測與辨識。

- **自動化環境檢查**：程式會確認 Hugging Face Token（`HF_TOKEN`）是否設定，並檢查所需的 Python 套件版本與圖片檔案是否存在。
- **多模態輸入處理**：將圖片路徑與「Text Recognition:」指令封裝成模型可理解的消息格式。
- **模型載入與推論**：自動下載並載入預訓練的處理器（Processor）與模型，利用硬體加速（`device_map="auto"`）進行文字生成。
- **結果解碼**：將模型生成的 Token 序列解碼為人類可讀的文字內容並輸出。

---

## 2. 需求套件

執行此程式碼需要安裝以下核心套件：

| 套件 | 說明 |
|------|------|
| `torch` | 深度學習運算框架 |
| `transformers` | 用於載入 Hugging Face 預訓練模型與處理器 |
| `pathlib` | 處理圖片檔案路徑 |
| `os` & `sys` | 用於系統環境變數設定與環境資訊輸出 |

---

## 3. 如何執行程式碼

請依照以下步驟在終端機（如 PowerShell）中執行：

1. **設定環境變數**
   為了下載受限或私有模型，請先設定您的 Hugging Face Token：
   ```powershell
   $env:HF_TOKEN="your_huggingface_token"
   ```

2. **執行 Python 腳本**
   使用您的虛擬環境執行主程式（假設檔名為 `ollama_test.py`）：
   ```powershell
   python ollama_test.py
   ```

3. **確認圖片路徑**
   程式預設會讀取位於以下路徑的圖片：
   ```
   C:/Users/Admin/Desktop/新增資料夾/圖檔/page1.png
   ```
   請確保該路徑下有對應檔案。

---

## 4. 輸出結果

程式執行後會先在控制台顯示環境資訊，最後輸出圖片辨識出的文字內容。

**辨識結果範例：**
![glm-ocr 分析](https://hackmd.io/_uploads/rJJcSxWEGe.png)![glm-ocr 分析 2](https://hackmd.io/_uploads/Byw5rxZNzg.png)



**原始輸入圖片參考：**

![請在此處放入您的 page1.png 圖片]

---

## 補充說明

根據程式碼顯示，模型會根據提供的圖片與 `Text Recognition:` 的指令，生成最大 `8192` 個 Token 的辨識結果。若未設定 Token，程式將嘗試以匿名方式存取模型。
