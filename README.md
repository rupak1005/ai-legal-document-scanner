# 📟 AI Legal Document Scanner

**AI Legal Document Scanner** is a powerful and intelligent web application that automates the understanding of scanned legal documents such as notices and summons. It leverages Optical Character Recognition (OCR), layout analysis, NLP, and transformers like LayoutLM and DistilBART to extract, classify, and summarize legal content—presented in an intuitive Streamlit interface.

---

Live link : https://ai-legal-document-scanner.streamlit.app/

##  Features

-  **Upload Scanned Legal Documents** (JPG, PNG)
-  **Text Extraction** using Tesseract OCR
-  **Document Classification** (Summons, Notice, Other)
-  **Layout-Aware Entity Detection** using LayoutLM
-  **Key-Value Pair Extraction** (form fields)
-  **JSON Output** of extracted entities and KV pairs
-  **Automated Document Summarization** using DistilBART
-  **Labeled Entity Visualization** on document image
-  **Download Options** for extracted data

---

##  Demo UI

The app is built using [Streamlit](https://streamlit.io/) and includes a clean UI that:

- Shows the scanned image with bounding boxes around detected text
- Displays classified document type and key-value fields
- Provides summarized content for quick comprehension
- Allows downloading extracted data as PNG, TXT, and JSON

---

##  Use Cases

- Scanning and understanding legal notices and summons
- Auto-classifying and extracting structured data from legal forms
- Summarizing long legal texts for legal assistants or clerks
- Digitizing law firm archives efficiently

---

##  Tech Stack

- **Frontend**: Streamlit (Python-based UI framework)
- **OCR**: Tesseract
- **Entity Recognition**: LayoutLM (HuggingFace Transformers)
- **Summarization**: DistilBART
- **NLP Tools**: Transformers, PyTorch
- **Image Handling**: OpenCV, PIL

---

##  Project Structure

```
.
├── app.py                    # Streamlit web app logic
├── inference.py             # Layout analysis, classification, summarization
├── utils/
│   ├── ocr.py               # OCR and text extraction utilities
│   └── preprocess.py        # Image preprocessing helpers
```

---

##  Installation

### 1. Clone the Repository
```bash
git clone https://github.com/rupak1005/ai-legal-document-scanner.git
cd ai-legal-document-scanner
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

> Make sure Tesseract is installed and added to PATH: https://github.com/tesseract-ocr/tesseract

---

##  Run the Application

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`

---

##  Sample Output

- **Text Detection Preview**
- **Labeled Entities Image**
- **Document Type** (Summons, Legal Notice, Other)
- **Extracted Fields** (Key-Value pairs)
- **Summary of Content**
- **Downloadable JSON/TXT/PNG files**

---

##  License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

##  Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Streamlit](https://streamlit.io/)

---

##  Future Enhancements

- Add table extraction from scanned legal docs
- Integrate a backend for saving and managing document history
- Enable support for PDFs
- Use fine-tuned legal-specific models for classification

---

##  Contributions

Contributions are welcome! Feel free to fork the repo, open issues, or submit PRs.

---

Made with ❤️ by [@rupak1005](https://github.com/rupak1005)
