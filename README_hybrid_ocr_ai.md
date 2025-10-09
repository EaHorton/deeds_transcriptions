# Hybrid OCR + AI Correction Script (`hybrid_ocr_ai.py`)

A cost-effective two-stage processing script that combines traditional Tesseract OCR with AI-powered text correction to provide accurate transcriptions of historical deed documents at a lower cost than pure AI vision processing.

## Overview

This script uses a hybrid approach:
1. **Stage 1**: Extract text using Tesseract OCR (fast and free)
2. **Stage 2**: Correct OCR errors using OpenAI's language models (cost-effective)

This approach provides high-quality results while significantly reducing API costs compared to pure AI vision processing, making it ideal for large document collections.

## Features

- **Two-Stage Processing**: Tesseract OCR followed by AI correction
- **Cost-Effective**: Dramatically lower costs than pure AI vision approach
- **High Accuracy**: Combines OCR speed with AI intelligence
- **Historical Document Expertise**: Specialized prompts for 1800s legal documents
- **Confidence Scoring**: Tesseract confidence analysis for quality assessment
- **Detailed Reporting**: Comprehensive processing statistics and cost tracking
- **Error Handling**: Robust processing with fallback options
- **Preservation Focus**: Maintains historical context and legal terminology

## Prerequisites

1. **Python 3.7+** with pip
2. **Tesseract OCR** installed on system:
   ```bash
   # macOS
   brew install tesseract
   
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # Windows
   # Download installer from https://github.com/UB-Mannheim/tesseract/wiki
   ```
3. **OpenAI API Key** set as environment variable:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```
4. **Virtual Environment** (recommended):
   ```bash
   # Create virtual environment
   python3 -m venv .venv
   
   # Activate virtual environment
   source .venv/bin/activate
   
   # Install dependencies
   pip install -r requirements_ocr_ai.txt
   ```

## Installation

Install Python dependencies:
```bash
pip install -r requirements_ocr_ai.txt
```

Required packages:
- `pytesseract` - Python wrapper for Tesseract OCR
- `openai` - OpenAI API client
- `Pillow` - Image processing

## Usage

### Basic Usage

```bash
# Using virtual environment (recommended)
./.venv/bin/python hybrid_ocr_ai.py

# Using activated virtual environment
source .venv/bin/activate
python hybrid_ocr_ai.py
deactivate
```

### Folder Structure Requirements

The script processes images in the `dupickens_b-1` folder:
```
dupickens/
└── dupickens_b-1/
    └── images/
        ├── dupickens_b-1_217.jpg
        ├── dupickens_b-1_218.jpg
        ├── dupickens_b-1_219.jpg
        └── ...
```

## Processing Workflow

### Stage 1: Tesseract OCR
- **Configuration**: PSM 3 (Fully automatic page segmentation)
- **Engine**: OEM 3 (Default, based on available engines)
- **Confidence Analysis**: Per-word confidence scoring
- **Text Cleaning**: Basic whitespace and formatting cleanup
- **Speed**: Very fast, processes images in seconds

### Stage 2: AI Correction
- **Model**: GPT-4o-mini (cost-optimized) or GPT-4o (higher quality)
- **Specialized Prompts**: Tailored for historical legal documents
- **Error Correction**: Fixes common OCR mistakes (0→O, 1→l, rn→m, etc.)
- **Context Preservation**: Maintains historical legal terminology
- **Format Enhancement**: Improves punctuation and readability

## Output Structure

The script creates organized output in the `ocr_ai_results/` directory:

```
ocr_ai_results/
├── dupickens_b-1_217_original_ocr.txt      ← Raw Tesseract output
├── dupickens_b-1_217_corrected.txt         ← AI-corrected text
├── dupickens_b-1_217_comparison.txt        ← Side-by-side comparison
├── dupickens_b-1_218_original_ocr.txt
├── dupickens_b-1_218_corrected.txt
├── dupickens_b-1_218_comparison.txt
├── ...
├── processing_summary.txt                  ← Detailed processing report
└── processing_summary.json                 ← JSON summary for analysis
```

### File Types Created

1. **Original OCR Files** (`*_original_ocr.txt`)
   - Raw Tesseract output with basic cleaning
   - Confidence scores and processing metadata
   - Unmodified OCR text for comparison

2. **Corrected Text Files** (`*_corrected.txt`)
   - AI-enhanced text with errors corrected
   - Improved formatting and punctuation
   - Historical context preserved

3. **Comparison Files** (`*_comparison.txt`)
   - Side-by-side view of original vs corrected text
   - Confidence scores and improvement analysis
   - Processing statistics for each stage

4. **Summary Files** (`processing_summary.*`)
   - Overall processing statistics
   - Cost breakdown and token usage
   - Performance metrics and timing analysis

## Example Output

### Console Output
```
Starting Hybrid OCR + AI Correction processing...
Images directory: /Users/eahorton/Downloads/deeds_unbound/dupickens/dupickens_b-1/images
Output directory: /Users/eahorton/Downloads/deeds_unbound/ocr_ai_results

Processing images with Tesseract OCR + AI correction...

Processing 1/8: dupickens_b-1_217.jpg
  Step 1: Extracting text with Tesseract OCR...
  ✓ OCR completed: 1,847 characters, 78.5% confidence (2.3s)
  Step 2: Correcting text with AI...
  ✓ AI correction completed (8.7s)
  Files saved: original_ocr.txt, corrected.txt, comparison.txt

Processing 2/8: dupickens_b-1_218.jpg
  Step 1: Extracting text with Tesseract OCR...
  ✓ OCR completed: 1,203 characters, 82.1% confidence (1.9s)
  Step 2: Correcting text with AI...
  ✓ AI correction completed (6.2s)
  Files saved: original_ocr.txt, corrected.txt, comparison.txt

...

========================================
Processing Complete!
========================================
Total images processed: 8
Total processing time: 89.7 seconds
OCR processing time: 16.4 seconds
AI correction time: 73.3 seconds
Total characters extracted: 14,239
Average OCR confidence: 79.8%
Total tokens used: 8,247
Total estimated cost: $0.157 USD
```

### Comparison File Example
```
HYBRID OCR + AI CORRECTION COMPARISON
=====================================
Source Image: dupickens_b-1_217.jpg
Processing Date: 2025-10-09 15:42:18
OCR Confidence: 78.5%
Processing Time: OCR: 2.3s, AI: 8.7s

ORIGINAL OCR TEXT:
------------------
V                    195
trustees shall be app0inted & c0nfirmed, shall be f0rthwith & f0rever fr0m 
c0mpleat discharge. 1n Witness where0f the said parties have t0 these 
presents interchangably set their hands and seals 0n the day...

CORRECTED TEXT:
---------------
V                    195
trustees shall be appointed & confirmed, shall be forthwith & forever from 
complete discharge. In Witness whereof the said parties have to these 
presents interchangeably set their hands and seals on the day...
```

## Cost Analysis

### Cost Comparison (Per Image)
- **Hybrid Approach**: ~$0.02 USD per image
- **Pure AI Vision**: ~$0.06 USD per image
- **Savings**: ~67% cost reduction while maintaining high quality

### Token Usage Breakdown
- **Input Tokens**: Raw OCR text (typically 500-2000 tokens)
- **Output Tokens**: Corrected text (similar length to input)
- **Model Used**: GPT-4o-mini (most cost-effective for text correction)

## Advanced Features

### OCR Configuration
- **PSM 3**: Fully automatic page segmentation without orientation detection
- **OEM 3**: Uses available OCR engines for best compatibility
- **Confidence Scoring**: Word-level confidence analysis
- **Image Preprocessing**: Automatic image optimization for OCR

### AI Correction Prompts
- **Historical Context**: Specialized knowledge of 1800s legal documents
- **Error Pattern Recognition**: Trained to recognize common OCR mistakes
- **Legal Terminology**: Preserves period-appropriate legal language
- **Format Enhancement**: Improves readability while maintaining authenticity

### Quality Assessment
- **Confidence Tracking**: Monitors OCR accuracy per image
- **Character Count Analysis**: Tracks text extraction completeness
- **Processing Time Metrics**: Performance monitoring for both stages
- **Cost Per Character**: Efficiency analysis

## Troubleshooting

### Common Issues

1. **Tesseract Not Found**
   ```bash
   pytesseract.pytesseract.TesseractNotFoundError
   ```
   **Solution**: Install Tesseract OCR system package
   ```bash
   # macOS
   brew install tesseract
   
   # Verify installation
   tesseract --version
   ```

2. **Low OCR Confidence**
   ```bash
   Warning: Low OCR confidence (45.2%) for image.jpg
   ```
   **Solution**: Image quality issues - consider image preprocessing or pure AI vision approach

3. **Module Import Errors**
   ```bash
   ModuleNotFoundError: No module named 'pytesseract'
   ```
   **Solution**: Use virtual environment Python
   ```bash
   ./.venv/bin/python hybrid_ocr_ai.py
   ```

4. **API Key Errors**
   ```bash
   Error: OPENAI_API_KEY environment variable not set
   ```
   **Solution**: Set your OpenAI API key
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

### Performance Optimization

- **Batch Processing**: Processes multiple images efficiently
- **Model Selection**: Uses cost-optimized GPT-4o-mini by default
- **Image Optimization**: Automatic image preprocessing for better OCR
- **Memory Management**: Efficient handling of large text documents

## When to Use This Script

### Best For:
- **Large document collections** where cost is a concern
- **Good quality images** with clear text
- **Historical documents** with standard layouts
- **Budget-conscious projects** requiring high accuracy

### Consider Alternatives For:
- **Poor quality images** → Use `ai_vision_ocr.py` (pure AI vision)
- **Complex layouts** → Use `ai_vision_ocr.py` (pure AI vision)
- **Single images** → Use `single_image_ai_ocr.py`
- **Traditional OCR only** → Use `ocr_processor.py`

## Technical Specifications

- **Supported Image Formats**: JPG, JPEG, PNG, TIFF, BMP
- **OCR Engine**: Tesseract 4.0+ with PSM 3 configuration
- **AI Model**: GPT-4o-mini (default) or GPT-4o
- **Token Limits**: 4,000 token output limit per correction
- **Processing Speed**: ~10-15 seconds per image (OCR + AI)
- **Cost Range**: $0.01-0.03 USD per image depending on text length

## Related Scripts

- **`ai_vision_ocr.py`**: Pure AI vision processing (higher cost, better quality)
- **`single_image_ai_ocr.py`**: Individual image processing with AI vision
- **`ocr_processor.py`**: Traditional Tesseract-only processing (free but lower accuracy)

## Support and Documentation

For detailed information:
- [Tesseract OCR Documentation](https://tesseract-ocr.github.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [OpenAI Pricing](https://openai.com/pricing)

## Version History

- **v1.0**: Initial hybrid processing implementation
- **v1.1**: Enhanced historical document prompts and error handling
- **v1.2**: Added comprehensive reporting and cost tracking
- **v1.3**: Optimized model selection and performance improvements