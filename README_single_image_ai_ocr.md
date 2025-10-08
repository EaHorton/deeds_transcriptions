# Single Image AI Vision OCR Script

This script processes a single specified image file using OpenAI's GPT-4 Vision API to extract text while preserving layout, structure, and handling multiple languages.

## Key Differences from ai_vision_ocr.py

- **Single Image Processing**: Processes one specific image instead of batch processing all folders
- **Command Line Interface**: Takes image path as a command line argument
- **Flexible Output**: Allows specifying custom output directory
- **Individual Processing**: Perfect for testing single documents or spot processing
- **Detailed Per-Image Summary**: Provides focused cost and usage tracking for single image

## Usage

**Important**: This script requires the OpenAI package which is installed in the virtual environment. Use one of these methods:

### Method 1: Direct Virtual Environment Usage (Recommended)
```bash
# From the project directory:
./.venv/bin/python single_image_ai_ocr.py <image_path> [output_dir]
```

### Method 2: Activate Virtual Environment First
```bash
# Activate the virtual environment:
source .venv/bin/activate

# Then run normally:
python single_image_ai_ocr.py <image_path> [output_dir]

# Deactivate when done:
deactivate
```

### Arguments

- `image_path` - Path to the image file to process (required)
- `output_dir` - Optional output directory (default: ocr_results_ai_vision/[folder_name])

### Examples

```bash
# Process a single image using virtual environment (recommended)
./.venv/bin/python single_image_ai_ocr.py dupickens/dupickens_b-1/images/dupickens_b-1_217.jpg

# Process with custom output directory
./.venv/bin/python single_image_ai_ocr.py dupickens/dupickens_b-1/images/dupickens_b-1_217.jpg /custom/output/path

# Using activated virtual environment
source .venv/bin/activate
python single_image_ai_ocr.py dupickens/dupickens_b-1/images/dupickens_b-1_217.jpg
deactivate

# Show help
./.venv/bin/python single_image_ai_ocr.py help
```

## Prerequisites

1. **Python 3.7+** with pip
2. **OpenAI API Key** set as environment variable:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Installation

Install Python dependencies:
```bash
pip install -r requirements_ai_vision.txt
```

## Output

The script creates two files in the `ocr_results_ai_vision/[folder_name]/` directory:

1. **`[image_name]_transcription.txt`** - Clean transcribed text with metadata
2. **`[image_name]_summary.txt`** - Processing summary with token usage and costs

For example, processing `dupickens/dupickens_b-1/images/dupickens_b-1_217.jpg` creates:
- `ocr_results_ai_vision/dupickens_b-1/dupickens_b-1_217_transcription.txt`
- `ocr_results_ai_vision/dupickens_b-1/dupickens_b-1_217_summary.txt`

## Example Output Structure

```
dupickens/dupickens_b-1/images/
├── dupickens_b-1_217.jpg              # Original image
└── transcriptions/                     # Created automatically
    ├── dupickens_b-1_217_transcription.txt
    └── dupickens_b-1_217_summary.txt
```

## Sample Usage Session

```bash
$ python single_image_ai_ocr.py dupickens/dupickens_b-1/images/dupickens_b-1_217.jpg

Single Image AI Vision OCR
==============================
Processing: dupickens_b-1_217.jpg
Full path: /Users/.../dupickens/dupickens_b-1/images/dupickens_b-1_217.jpg

Extracting text with AI Vision...
Saving transcription...

==================================================
Processing Complete!
==================================================
Processing time: 12.3 seconds
Text length: 2,137 characters
Tokens used: 2,456
  - Input: 1,200
  - Output: 1,256
Estimated cost: $0.0247 USD

Files created:
  Transcription: /Users/.../transcriptions/dupickens_b-1_217_transcription.txt
  Summary: /Users/.../transcriptions/dupickens_b-1_217_summary.txt

Text preview:
V                                                                                      195
trustees shall be appointed & confirmed, shall be forthwit...
```

## Features

- **High-Quality Transcription**: Uses GPT-4o for optimal accuracy
- **Layout Preservation**: Maintains original document structure and formatting
- **Multi-Language Support**: Handles non-Latin scripts (Arabic, Chinese, etc.)
- **Cost Tracking**: Detailed token usage and cost reporting
- **Error Handling**: Robust processing with informative error messages
- **Flexible Output**: Custom output directory support
- **Image Optimization**: Automatically resizes large images for API efficiency

## Supported Image Formats

- `.jpg`, `.jpeg`
- `.png`
- `.tiff`
- `.bmp`
- `.webp`

## Error Handling

The script provides clear error messages for common issues:

- Missing image file
- Unsupported file format
- Missing OpenAI API key
- Invalid file paths
- API errors

## Cost Considerations

- **GPT-4o pricing**: ~$0.005 per 1K input tokens, ~$0.015 per 1K output tokens
- **Typical cost per image**: $0.01 - $0.05 USD depending on image complexity
- **Processing time**: 10-30 seconds per image depending on content

## When to Use This vs ai_vision_ocr.py

**Use single_image_ai_ocr.py when:**
- Testing transcription quality on specific documents
- Processing individual files on demand
- Need custom output locations
- Debugging transcription issues
- Processing files outside the standard folder structure

**Use ai_vision_ocr.py when:**
- Batch processing entire collections
- Processing all documents in dupickens folders
- Need organized folder-based output structure
- Running large-scale document processing

## Troubleshooting

1. **"Image file not found"**: Check file path and ensure file exists
2. **"Unsupported file format"**: Convert image to supported format (JPG, PNG, etc.)
3. **"OPENAI_API_KEY not set"**: Set environment variable with your API key
4. **API errors**: Check internet connection and API key validity
5. **High costs**: Consider using smaller images or gpt-4o-mini model (edit script)

## Customization

To use gpt-4o-mini instead of gpt-4o (lower cost, potentially lower quality):

Edit line 371 in the script:
```python
# Change this:
transcribed_text = extract_text_with_ai_vision(image_path, client, tracker, model="gpt-4o")

# To this:
transcribed_text = extract_text_with_ai_vision(image_path, client, tracker, model="gpt-4o-mini")
```