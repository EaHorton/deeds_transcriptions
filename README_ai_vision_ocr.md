# AI Vision OCR Script (`ai_vision_ocr.py`)

A comprehensive batch processing script that uses OpenAI's GPT-4 Vision API to extract text from historical deed documents while preserving original layout and structure.

## Overview

This script automatically discovers and processes all images in the `dupickens` subfolder structure, using advanced AI vision to accurately transcribe historical documents without interpretation or translation. It provides detailed cost tracking and creates organized output with comprehensive processing summaries.

## Features

- **Batch Processing**: Automatically processes all dupickens subfolders with images
- **AI-Powered OCR**: Uses GPT-4 Vision for high-accuracy text extraction
- **Layout Preservation**: Maintains original document structure and spacing
- **Multi-Language Support**: Preserves original text in any language without translation
- **Cost Tracking**: Detailed token usage and cost analysis
- **Organized Output**: Structured results with transcriptions and summaries separated
- **Error Handling**: Robust processing with detailed error reporting
- **Progress Tracking**: Real-time updates during batch processing

## Prerequisites

1. **Python 3.7+** with pip
2. **OpenAI API Key** set as environment variable:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```
3. **Virtual Environment** (recommended):
   ```bash
   # Create virtual environment
   python3 -m venv .venv
   
   # Activate virtual environment
   source .venv/bin/activate
   
   # Install dependencies
   pip install -r requirements_ai_vision.txt
   ```

## Installation

Install Python dependencies:
```bash
pip install -r requirements_ai_vision.txt
```

Required packages:
- `openai` - OpenAI API client
- `Pillow` - Image processing

## Usage

### Basic Usage

```bash
# Using virtual environment (recommended)
./.venv/bin/python ai_vision_ocr.py

# Using activated virtual environment
source .venv/bin/activate
python ai_vision_ocr.py
deactivate
```

### Folder Structure Requirements

The script expects this folder structure:
```
dupickens/
├── dupickens_b-1/
│   └── images/
│       ├── dupickens_b-1_217.jpg
│       ├── dupickens_b-1_218.jpg
│       └── ...
├── dupickens_b-2/
│   └── images/
│       └── ...
├── dupickens_c-1/
│   └── images/
│       └── ...
└── ...
```

## Output Structure

The script creates organized output in the `ocr_results_ai_vision/` directory:

```
ocr_results_ai_vision/
├── summaries/                           ← Main processing summaries
│   ├── usage_summary.txt               ← Detailed text summary
│   └── usage_summary.json              ← JSON summary for analysis
├── dupickens_b-1/
│   ├── dupickens_b-1_217_transcription.txt
│   ├── dupickens_b-1_218_transcription.txt
│   └── summaries/                       ← Individual processing summaries
│       ├── dupickens_b-1_217_summary.txt
│       └── dupickens_b-1_218_summary.txt
├── dupickens_b-2/
│   ├── [transcription files]
│   └── summaries/
└── [other dupickens folders]/
```

### File Types Created

1. **Transcription Files** (`*_transcription.txt`)
   - Complete extracted text with original layout preserved
   - Source image metadata and processing timestamp
   - Clean, readable format suitable for analysis

2. **Individual Summary Files** (`summaries/*_summary.txt`)
   - Processing details for each image
   - Token usage and cost breakdown
   - Processing time and model information

3. **Batch Summary Files** (`summaries/usage_summary.*`)
   - Overall processing statistics
   - Total costs and token usage across all images
   - Summary of all processed files

## Processing Details

### AI Vision Processing
- **Model**: GPT-4 Vision (gpt-4o for optimal cost/performance)
- **Image Optimization**: Automatically resizes large images for API efficiency
- **Layout Preservation**: Maintains original document structure
- **Language Handling**: Preserves original text without translation
- **Quality Settings**: High-detail processing for maximum accuracy

### Cost Management
- **Real-time Tracking**: Monitor costs during processing
- **Detailed Breakdown**: Per-image and total cost analysis
- **Token Monitoring**: Input/output token usage tracking
- **Model Optimization**: Uses cost-effective gpt-4o model by default

## Example Output

### Console Output
```
Starting AI Vision OCR processing...
Base directory: /Users/eahorton/Downloads/deeds_unbound
Base output directory: /Users/eahorton/Downloads/deeds_unbound/ocr_results_ai_vision

Found 4 dupickens folders to process:
  - dupickens/dupickens_b-1 (8 images)
  - dupickens/dupickens_c-1 (12 images)
  - dupickens/dupickens_d-1 (15 images)
  - dupickens/dupickens_e-1 (6 images)

Processing dupickens_b-1 (8 images)...
  1/8: dupickens_b-1_217.jpg... ✓ (15.2s, 2,128 chars, $0.012)
  2/8: dupickens_b-1_218.jpg... ✓ (18.7s, 1,847 chars, $0.011)
  ...

Total processing time: 45.3 minutes
Total images processed: 41
Total cost: $2.47 USD
```

### Transcription File Example
```
Source Image: dupickens_b-1_217.jpg
Full Path: /Users/.../dupickens/dupickens_b-1/images/dupickens_b-1_217.jpg
Transcription Date: 2025-10-09 14:23:15
============================================================

V                    195

trustees shall be appointed & confirmed, shall be forthwith & forever from 
compleat discharge. In Witness whereof the said parties have to these 
presents interchangably set their hands and seals on the day...
```

## Advanced Features

### Error Handling
- **Image Processing Errors**: Graceful handling of corrupted files
- **API Errors**: Retry logic and detailed error reporting
- **File System Errors**: Comprehensive path and permission checking

### Performance Optimization
- **Image Resizing**: Automatic optimization for API limits
- **Batch Processing**: Efficient folder traversal and processing
- **Memory Management**: Optimized handling of large image collections

### Monitoring and Reporting
- **Progress Indicators**: Real-time processing updates
- **Cost Estimation**: Continuous cost tracking during processing
- **Quality Metrics**: Character count and processing time per image

## Troubleshooting

### Common Issues

1. **Module Import Errors**
   ```bash
   ModuleNotFoundError: No module named 'openai'
   ```
   **Solution**: Use virtual environment Python
   ```bash
   ./.venv/bin/python ai_vision_ocr.py
   ```

2. **API Key Errors**
   ```bash
   Error: OPENAI_API_KEY environment variable not set
   ```
   **Solution**: Set your OpenAI API key
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

3. **No Images Found**
   ```bash
   No dupickens subfolders with images directories found
   ```
   **Solution**: Check folder structure matches expected format

### Cost Management

- **Monitor Usage**: Check costs regularly during large batch processing
- **Rate Limits**: Script includes automatic rate limiting for API calls
- **Budget Alerts**: Consider setting OpenAI account spending limits

## Technical Specifications

- **Supported Image Formats**: JPG, JPEG, PNG, WEBP
- **Maximum Image Size**: Automatically optimized to 2048x2048 pixels
- **API Model**: GPT-4 Vision (gpt-4o recommended)
- **Token Limits**: 4,000 token output limit per image
- **Processing Speed**: ~15-30 seconds per image depending on complexity

## Related Scripts

- **`single_image_ai_ocr.py`**: Process individual images
- **`hybrid_ocr_ai.py`**: Tesseract + AI correction approach
- **`ocr_processor.py`**: Traditional Tesseract-only processing

## Support and Documentation

For detailed API documentation and current pricing, see:
- [OpenAI Vision API Documentation](https://platform.openai.com/docs/guides/vision)
- [OpenAI Pricing](https://openai.com/pricing)

## Version History

- **v1.0**: Initial batch processing implementation
- **v1.1**: Added organized output structure with summaries subfolder
- **v1.2**: Enhanced error handling and progress reporting