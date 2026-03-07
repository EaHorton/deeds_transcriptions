# Single Image Claude Vision OCR

This script uses Anthropic's Claude Vision API to extract text from individual document images. Claude Vision is particularly well-suited for historical documents with sensitive content, as it generally has more permissive content policies for archival and academic research purposes compared to GPT-4 Vision.

## Key Features

- **Better Handling of Historical Documents**: Claude Vision is less likely to refuse processing documents containing sensitive historical content (slavery records, racial language, etc.)
- **High Accuracy OCR**: Maintains layout, structure, and formatting
- **Hallucination Detection**: Integrated validation to detect and clean AI hallucinations
- **Multi-language Support**: Preserves original text in any language or script
- **Token Tracking**: Monitors usage and calculates costs
- **Quality Validation**: Checks image quality, text structure, and document elements

## Use Cases

This script is ideal for:
- Historical slavery and property records
- Documents that get blocked by GPT-4 Vision's content filters
- Archival preservation projects
- Genealogical research documents
- Any sensitive historical material requiring accurate transcription

## Installation

1. Install required packages:
```bash
pip install -r requirements_claude_vision.txt
```

Or install manually:
```bash
pip install anthropic pillow
```

2. Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

To get an API key:
- Sign up at https://console.anthropic.com/
- Navigate to API Keys section
- Generate a new API key

## Usage

### Basic Usage

Process a single image:
```bash
python single_image_claude_ocr.py path/to/image.jpg
```

### Using Virtual Environment

```bash
# Activate virtual environment
source .venv/bin/activate

# Process an image
python single_image_claude_ocr.py dupickens/dupickens_b-1/images/dupickens_b-1_423.jpg
```

Or without activating:
```bash
./.venv/bin/python single_image_claude_ocr.py dupickens/dupickens_c-1/images/dupickens_c-1_050.jpg
```

### Custom Output Directory

```bash
python single_image_claude_ocr.py image.jpg /custom/output/path
```

## Output Structure

The script creates two files:

### 1. Transcription File
Located at: `ocr_results_claude_vision/[folder_name]/[image_name]_transcription.txt`

Contains:
- Source image metadata
- Validation status
- Image quality metrics (DPI)
- Complete transcribed text

### 2. Summary File
Located at: `ocr_results_claude_vision/[folder_name]/summaries/[image_name]_summary.txt`

Contains:
- Processing time
- Token usage (input/output)
- Estimated cost
- Model details
- Hallucination detection results

## Supported Image Formats

- `.jpg` / `.jpeg`
- `.png`
- `.gif`
- `.webp`

## Hallucination Detection

The script automatically:
1. Analyzes extracted text for potential hallucinations
2. Detects repeated text blocks
3. Validates document structure
4. Checks image quality (DPI)
5. Cleans duplicated content if detected

## Pricing

Claude 3.5 Sonnet (recommended model):
- Input: $3 per 1M tokens
- Output: $15 per 1M tokens

Typical cost per document image: $0.01 - $0.05

## Comparison: Claude vs GPT-4 Vision

| Feature | Claude Vision | GPT-4 Vision |
|---------|--------------|--------------|
| Historical sensitive content | ✅ Better | ⚠️ Often blocked |
| Cost per image | ~$0.02 | ~$0.03 |
| Accuracy | Excellent | Excellent |
| Content policy flexibility | More permissive | More restrictive |
| Max output tokens | 4096 | 4000 |

## When to Use Claude vs GPT-4

**Use Claude (`single_image_claude_ocr.py`) when:**
- Documents contain slavery-related content
- GPT-4 Vision refuses to process the document
- Document contains sensitive historical racial language
- Academic/archival preservation work

**Use GPT-4 (`single_image_ai_ocr.py`) when:**
- Standard historical documents without sensitive content
- General property deeds and legal documents
- Non-controversial historical records

## Troubleshooting

### API Key Not Set
```
Error: ANTHROPIC_API_KEY environment variable not set.
```

Solution:
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

### Image Not Found
```
Error: Image file not found: path/to/image.jpg
```

Solution: Verify the image path is correct and the file exists.

### Validation Warnings

If you see validation warnings like "Missing expected elements", this indicates the document may not be a standard South Carolina legal document, or OCR quality may need review. The transcription is still saved and should be manually reviewed.

## Example Workflow

For the problematic documents identified:

```bash
# Set API key (one time per terminal session)
export ANTHROPIC_API_KEY='your-key'

# Process blocked documents with Claude
python single_image_claude_ocr.py dupickens/dupickens_b-1/images/dupickens_b-1_423.jpg
python single_image_claude_ocr.py dupickens/dupickens_c-1/images/dupickens_c-1_050.jpg
python single_image_claude_ocr.py dupickens/dupickens_c-1/images/dupickens_c-1_060.jpg
python single_image_claude_ocr.py dupickens/dupickens_c-1/images/dupickens_c-1_076.jpg
python single_image_claude_ocr.py dupickens/dupickens_d-1/images/dupickens_d-1_137.jpg
```

## Technical Details

- **Model**: claude-3-5-sonnet-20241022 (default)
- **Temperature**: 0 (deterministic output)
- **Max tokens**: 4096
- **Image optimization**: Automatic resizing to 2048x2048 max
- **Encoding**: UTF-8 for all output files

## Related Scripts

- `single_image_ai_ocr.py` - GPT-4 Vision version
- `ai_vision_ocr.py` - Batch processing with GPT-4 Vision
- `hybrid_ocr_ai.py` - Tesseract + AI correction
- `hallucination_detector.py` - Validation module

## License

This script is part of the Deeds Unbound historical document preservation project.
