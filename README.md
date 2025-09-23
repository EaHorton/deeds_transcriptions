# OCR Processing Script

This script processes images in the `dupickens/dupickens_b-1/images/` directory using Tesseract OCR with PSM 3 (Fully automatic page segmentation).

## Features

- **Automatic Image Processing**: Loops through all images in the target directory
- **PSM 3 Configuration**: Uses Tesseract's Page Segmentation Mode 3 for optimal text detection
- **Text Cleaning**: Removes OCR artifacts and normalizes whitespace
- **Confidence Scoring**: Calculates average confidence scores for each image
- **Multiple Output Formats**: Saves results as both text files and JSON data
- **Summary Report**: Generates an overview of all processed images

## Prerequisites

1. **Python 3.7+** with pip
2. **Tesseract OCR** binary installed on your system:
   - **macOS**: `brew install tesseract`
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

## Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Verify Tesseract is installed:
   ```bash
   tesseract --version
   ```

## Usage

Run the script from the project directory:

```bash
python ocr_processor.py
```

## Output

The script creates an `ocr_results/` directory containing:

- **Individual text files** (`*_ocr.txt`): Clean text with confidence scores
- **JSON data files** (`*_data.json`): Complete OCR data including coordinates
- **Summary report** (`ocr_summary.txt`): Overview of all processed images

## Sample Output

```
OCR Processing Summary
==============================

Total Images Processed: 4
Average Confidence Score: 57.58%

Individual Results:
--------------------
dupickens_b-1_217.jpg: 47.30% confidence
dupickens_b-1_218.jpg: 69.29% confidence
dupickens_b-1_219.jpg: 66.98% confidence
dupickens_b-1_240.jpg: 46.75% confidence
```

## Text Cleaning Features

The script automatically:
- Removes excessive whitespace and normalizes line breaks
- Filters out common OCR artifacts
- Fixes common character recognition errors (0→O, l→I, rn→m)
- Preserves only alphanumeric characters and common punctuation

## Configuration

To modify OCR settings, edit the `custom_config` variable in the `process_image()` function:

```python
custom_config = r'--oem 3 --psm 3'
```

Available PSM modes:
- PSM 3: Fully automatic page segmentation (default)
- PSM 6: Uniform block of text
- PSM 8: Single word
- PSM 13: Raw line (used for single text line)

## Troubleshooting

1. **"tesseract not found"**: Ensure Tesseract is installed and in your PATH
2. **Low confidence scores**: Consider image preprocessing (contrast, rotation, noise reduction)
3. **No text detected**: Try different PSM modes or check image quality

## File Structure

```
project/
├── ocr_processor.py      # Main script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── dupickens/           # Source images
│   └── dupickens_b-1/
│       └── images/
└── ocr_results/         # Generated output
    ├── *_ocr.txt       # Clean text files
    ├── *_data.json     # Complete OCR data
    └── ocr_summary.txt # Processing summary
```