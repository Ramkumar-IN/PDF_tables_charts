# PDF Chart & Table Extractor

This project extracts **charts** and **tables** from PDF files using YOLO-based detection and classification models.

---

## Installation

Make sure you have **Python 3.8+** installed. Then install the required Python packages:

```bash
pip install -r requirements.txt
Additionally, install Poppler (required by pdf2image):

bash
Copy code
sudo apt-get update
sudo apt-get install -y poppler-utils
Usage
Place your PDF files in the input_pdfs/ folder.

Make sure the YOLO models (.pt files) are in the models/ folder.

Run the script:

bash
Copy code
python extractor_charts_tables.py
The extracted charts and tables will be saved in output_charts/ and output_tables/ folders respectively.

File Structure
Copy code
pdf_chart_tables_extractor/
├── models/
│   ├── doclayout_yolo_docstructbench_imgsz1024.pt
│   └── classification_chart.pt
├── input_pdfs/
│   └── your_pdf_files.pdf
├── output_charts/
├── output_tables/
├── extractor_charts_tables.py
├── requirements.txt
└── README.md
Citation
Detection model adapted from DocLayout-YOLO-DocStructBench.

License
This project is for research and educational purposes. Make sure to comply with the licenses of the included models.
