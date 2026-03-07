#!/usr/bin/env python3
"""
Single Image Claude Vision OCR Script
Uses Anthropic's Claude Vision to extract text from a single specified image while preserving layout,
structure, and handling multiple languages. Claude generally has better handling of historical documents
with sensitive content compared to GPT-4 Vision.

Usage:
    # Using virtual environment (recommended):
    /path/to/.venv/bin/python single_image_claude_ocr.py path/to/image.jpg
    
    # Or activate virtual environment first:
    source .venv/bin/activate
    python single_image_claude_ocr.py path/to/image.jpg
"""

import os
import sys
import base64
import json
import time
from pathlib import Path
from typing import Dict, Tuple
from PIL import Image
import anthropic
from hallucination_detector import HallucinationDetector

# Claude API pricing (as of 2024 - verify current rates)
PRICING = {
    'claude-3-5-sonnet-20241022': {
        'input_tokens': 0.003 / 1000,      # $3 per 1M tokens
        'output_tokens': 0.015 / 1000,     # $15 per 1M tokens
    },
    'claude-3-opus-20240229': {
        'input_tokens': 0.015 / 1000,      # $15 per 1M tokens
        'output_tokens': 0.075 / 1000,     # $75 per 1M tokens
    },
    'claude-3-sonnet-20240229': {
        'input_tokens': 0.003 / 1000,      # $3 per 1M tokens
        'output_tokens': 0.015 / 1000,     # $15 per 1M tokens
    }
}

class TokenTracker:
    """Track token usage and calculate costs for Claude API calls."""
    
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.model_usage = {}
        self.processing_time = 0
    
    def add_usage(self, model: str, input_tokens: int, output_tokens: int):
        """Add token usage for a specific model."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1
        
        if model not in self.model_usage:
            self.model_usage[model] = {'input': 0, 'output': 0, 'requests': 0}
        
        self.model_usage[model]['input'] += input_tokens
        self.model_usage[model]['output'] += output_tokens
        self.model_usage[model]['requests'] += 1
    
    def calculate_cost(self) -> float:
        """Calculate total cost based on token usage."""
        total_cost = 0.0
        
        for model, usage in self.model_usage.items():
            if model in PRICING:
                model_cost = (
                    usage['input'] * PRICING[model]['input_tokens'] +
                    usage['output'] * PRICING[model]['output_tokens']
                )
                total_cost += model_cost
        
        return total_cost
    
    def get_summary(self) -> Dict:
        """Get detailed usage and cost summary."""
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'total_requests': self.total_requests,
            'total_cost': self.calculate_cost(),
            'processing_time': self.processing_time,
            'model_breakdown': self.model_usage
        }

def encode_image_to_base64(image_path: str) -> Tuple[str, str]:
    """
    Encode image to base64 string for Claude API.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        Tuple[str, str]: (Base64 encoded image, media_type)
    """
    # Determine media type from file extension
    ext = Path(image_path).suffix.lower()
    media_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    media_type = media_type_map.get(ext, 'image/jpeg')
    
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode('utf-8'), media_type

def optimize_image_for_api(image_path: str, max_size: Tuple[int, int] = (2048, 2048)) -> str:
    """
    Optimize image size for API while maintaining quality.
    
    Args:
        image_path (str): Path to the original image
        max_size (Tuple[int, int]): Maximum dimensions (width, height)
        
    Returns:
        str: Path to optimized image (or original if no optimization needed)
    """
    try:
        with Image.open(image_path) as img:
            # Check if image needs resizing
            if img.size[0] <= max_size[0] and img.size[1] <= max_size[1]:
                return image_path
            
            # Calculate new size maintaining aspect ratio
            ratio = min(max_size[0] / img.size[0], max_size[1] / img.size[1])
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            
            # Resize and save optimized version
            optimized_img = img.resize(new_size, Image.Resampling.LANCZOS)
            optimized_path = image_path.replace('.jpg', '_optimized.jpg').replace('.jpeg', '_optimized.jpeg').replace('.png', '_optimized.png')
            
            # Convert to RGB if necessary for JPEG
            if optimized_img.mode != 'RGB' and optimized_path.lower().endswith(('.jpg', '.jpeg')):
                optimized_img = optimized_img.convert('RGB')
            
            optimized_img.save(optimized_path, quality=95, optimize=True)
            return optimized_path
            
    except Exception as e:
        print(f"Warning: Could not optimize image {image_path}: {e}")
        return image_path

def extract_text_with_claude_vision(image_path: str, client: anthropic.Anthropic, tracker: TokenTracker, 
                                    model: str = "claude-3-5-sonnet-20241022") -> str:
    """
    Extract text from image using Claude's Vision API.
    
    Args:
        image_path (str): Path to the image file
        client (anthropic.Anthropic): Anthropic client instance
        tracker (TokenTracker): Token usage tracker
        model (str): Model to use for vision processing
        
    Returns:
        str: Extracted text preserving original layout and language
    """
    try:
        start_time = time.time()
        
        # Optimize image for API
        optimized_path = optimize_image_for_api(image_path)
        
        # Encode image to base64
        base64_image, media_type = encode_image_to_base64(optimized_path)
        
        # Prepare the prompt for accurate transcription with historical context
        prompt = """You are assisting with historical archival work to preserve and digitize legal records from the 1800s. These documents are of significant historical importance for understanding American history, including slavery records, property transfers, and genealogical research.

TASK: Extract ALL text from this historical document image exactly as it appears.

CRITICAL GUIDELINES:
1. This is for academic historical preservation - transcribe faithfully without judgment or commentary
2. Preserve all original text including names, dates, and legal terminology from the 1800s
3. Maintain exact layout, line breaks, and spatial relationships
4. Do not add any commentary, interpretation, explanations, or moral judgments
5. Use [?] only for completely illegible characters
6. Preserve original spelling, punctuation, and formatting exactly as written
7. Include all headers, titles, dates, numbers, and visible text elements
8. If text is in a non-Latin script, transcribe it exactly as written - do not translate
9. Maintain document structure - use appropriate spacing and line breaks to match the original
10. If there are multiple columns or sections, preserve their structure

Return ONLY the verbatim transcribed text with no additional commentary, introduction, or conclusion."""

        # Make API call with Claude Vision
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
            temperature=0  # Use 0 for consistent, deterministic transcription
        )
        
        # Track token usage
        tracker.add_usage(model, message.usage.input_tokens, message.usage.output_tokens)
        
        # Track processing time
        processing_time = time.time() - start_time
        tracker.processing_time = processing_time
        
        # Clean up optimized image if it was created
        if optimized_path != image_path and os.path.exists(optimized_path):
            os.remove(optimized_path)
        
        # Extract text from response
        text_content = ""
        for block in message.content:
            if block.type == "text":
                text_content += block.text
        
        return text_content.strip()
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        # Clean up optimized image on error
        if 'optimized_path' in locals() and optimized_path != image_path and os.path.exists(optimized_path):
            os.remove(optimized_path)
        return f"[ERROR: Could not process image - {str(e)}]"

def save_transcription(image_path: str, transcribed_text: str, output_dir: str = None, 
                      validation_results: dict = None) -> str:
    """
    Save transcribed text to a file.
    
    Args:
        image_path (str): Path to the source image
        transcribed_text (str): Extracted text
        output_dir (str): Directory to save the transcription (optional)
        validation_results (dict): Hallucination detection results (optional)
        
    Returns:
        str: Path to saved file
    """
    image_path_obj = Path(image_path)
    
    # Determine output directory
    if output_dir is None:
        # Use ocr_results_claude_vision structure
        # Extract folder name from path (e.g., dupickens_b-1 from dupickens/dupickens_b-1/images/...)
        folder_name = None
        for part in image_path_obj.parts:
            if part.startswith('dupickens_'):
                folder_name = part
                break
        
        if folder_name:
            output_dir = Path("ocr_results_claude_vision") / folder_name
        else:
            # Fallback to old behavior if folder pattern not found
            output_dir = image_path_obj.parent / "transcriptions_claude"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename
    base_name = image_path_obj.stem
    output_file = output_dir / f"{base_name}_transcription.txt"
    
    # Save transcription
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Source Image: {image_path_obj.name}\n")
        f.write(f"Full Path: {image_path}\n")
        f.write(f"Transcription Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"AI Model: Claude Vision\n")
        
        # Add validation results if available
        if validation_results:
            f.write(f"Validation Status: {'⚠️  NEEDS REVIEW' if validation_results['needs_review'] else '✓ PASSED'}\n")
            if validation_results.get('quality_metrics', {}).get('dpi'):
                f.write(f"Image DPI: {validation_results['quality_metrics']['dpi']:.1f}\n")
            if validation_results['needs_review']:
                f.write(f"Review Reasons:\n")
                for reason in validation_results['review_reasons']:
                    f.write(f"  - {reason}\n")
        
        f.write("=" * 60 + "\n\n")
        f.write(transcribed_text)
    
    return str(output_file)

def save_usage_summary(tracker: TokenTracker, output_file: str, image_path: str, 
                      validation_results: dict = None) -> None:
    """
    Save detailed usage and cost summary.
    
    Args:
        tracker (TokenTracker): Token usage tracker
        output_file (str): Path to save the summary
        image_path (str): Path to the processed image
        validation_results (dict): Hallucination detection results (optional)
    """
    summary = tracker.get_summary()
    
    # Save summary as text
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Single Image Claude Vision OCR - Processing Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Image Processed: {Path(image_path).name}\n")
        f.write(f"Full Path: {image_path}\n")
        f.write(f"Processing Time: {summary['processing_time']:.2f} seconds\n\n")
        
        f.write("Token Usage:\n")
        f.write("-" * 12 + "\n")
        f.write(f"Input Tokens:  {summary['total_input_tokens']:,}\n")
        f.write(f"Output Tokens: {summary['total_output_tokens']:,}\n")
        f.write(f"Total Tokens:  {summary['total_tokens']:,}\n\n")
        
        f.write(f"Estimated Cost: ${summary['total_cost']:.4f} USD\n\n")
        
        if summary['model_breakdown']:
            f.write("Model Details:\n")
            f.write("-" * 14 + "\n")
            for model, usage in summary['model_breakdown'].items():
                model_cost = 0
                if model in PRICING:
                    model_cost = (
                        usage['input'] * PRICING[model]['input_tokens'] +
                        usage['output'] * PRICING[model]['output_tokens']
                    )
                f.write(f"Model: {model}\n")
                f.write(f"  Input tokens: {usage['input']:,}\n")
                f.write(f"  Output tokens: {usage['output']:,}\n")
                f.write(f"  Cost: ${model_cost:.4f}\n")
        
        # Add validation results if available
        if validation_results:
            f.write("\nHallucination Detection Results:\n")
            f.write("-" * 32 + "\n")
            f.write(f"Status: {'⚠️  NEEDS REVIEW' if validation_results['needs_review'] else '✓ PASSED'}\n")
            f.write(f"Text Length: {validation_results['text_length']:,} characters\n")
            
            if validation_results.get('quality_metrics', {}).get('dpi'):
                f.write(f"Image DPI: {validation_results['quality_metrics']['dpi']:.1f}\n")
            
            if validation_results['needs_review']:
                f.write(f"\nReview Required - Reasons:\n")
                for reason in validation_results['review_reasons']:
                    f.write(f"  • {reason}\n")
            else:
                f.write(f"\nNo issues detected. Text passed all validation checks.\n")

def print_usage():
    """Print usage instructions."""
    print("Single Image Claude Vision OCR")
    print("=" * 35)
    print()
    print("Usage:")
    print("  # Using virtual environment (recommended):")
    print("  /path/to/.venv/bin/python single_image_claude_ocr.py <image_path> [output_dir]")
    print()
    print("  # Or activate virtual environment first:")
    print("  source .venv/bin/activate")
    print("  python single_image_claude_ocr.py <image_path> [output_dir]")
    print()
    print("Arguments:")
    print("  image_path   Path to the image file to process")
    print("  output_dir   Optional output directory (default: ocr_results_claude_vision/[folder_name])")
    print()
    print("Examples:")
    print("  # From this project directory:")
    print("  ./.venv/bin/python single_image_claude_ocr.py dupickens/dupickens_b-1/images/dupickens_b-1_423.jpg")
    print("  ./.venv/bin/python single_image_claude_ocr.py image.jpg /custom/output")
    print()
    print("  # With activated virtual environment:")
    print("  source .venv/bin/activate")
    print("  python single_image_claude_ocr.py dupickens/dupickens_c-1/images/dupickens_c-1_050.jpg")
    print()
    print("Supported formats: .jpg, .jpeg, .png, .gif, .webp")
    print()
    print("Note: Claude Vision generally handles historical documents with")
    print("      sensitive content better than GPT-4 Vision.")

def main():
    """Main function to process a single image with Claude vision."""
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Error: No image file specified.")
        print()
        print_usage()
        sys.exit(1)
    
    if sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
        sys.exit(0)
    
    # Get image path from command line
    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Convert to absolute path if relative
    if not os.path.isabs(image_path):
        image_path = os.path.abspath(image_path)
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Check if it's a valid image file
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    if not Path(image_path).suffix.lower() in image_extensions:
        print(f"Error: Unsupported file format. Supported formats: {', '.join(image_extensions)}")
        sys.exit(1)
    
    # Check for API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Please set your Anthropic API key:")
        print("export ANTHROPIC_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Initialize Claude client, tracker, and hallucination detector
    client = anthropic.Anthropic(api_key=api_key)
    tracker = TokenTracker()
    detector = HallucinationDetector(max_length=20000, min_confidence=0.85, min_dpi=200)
    
    print("Single Image Claude Vision OCR")
    print("=" * 35)
    print(f"Processing: {Path(image_path).name}")
    print(f"Full path: {image_path}")
    if output_dir:
        print(f"Output directory: {output_dir}")
    print()
    
    # Process the image
    start_time = time.time()
    
    print("Extracting text with Claude Vision...")
    transcribed_text = extract_text_with_claude_vision(image_path, client, tracker, 
                                                       model="claude-3-5-sonnet-20241022")
    
    # Analyze for hallucinations
    print("Analyzing text for hallucinations...")
    validation_results = detector.analyze_text(transcribed_text, image_path)
    
    # Clean hallucinations if detected
    if validation_results['needs_review']:
        print(f"⚠️  Validation issues detected: {len(validation_results['review_reasons'])} issue(s)")
        cleaned_text = detector.clean_hallucinations(transcribed_text)
        if len(cleaned_text) != len(transcribed_text):
            print(f"   Removed {len(transcribed_text) - len(cleaned_text)} characters of duplicated content")
            transcribed_text = cleaned_text
    else:
        print("✓ Text validation passed")
    
    # Save transcription
    print("Saving transcription...")
    output_file = save_transcription(image_path, transcribed_text, output_dir, validation_results)
    
    # Save usage summary in dedicated summaries subfolder
    summary_dir = Path(output_file).parent / "summaries"
    summary_dir.mkdir(exist_ok=True)
    summary_file = summary_dir / f"{Path(image_path).stem}_summary.txt"
    save_usage_summary(tracker, str(summary_file), image_path, validation_results)
    
    # Display results
    end_time = time.time()
    processing_time = end_time - start_time
    summary = tracker.get_summary()
    
    print("\n" + "=" * 50)
    print("Processing Complete!")
    print("=" * 50)
    print(f"Processing time: {processing_time:.1f} seconds")
    print(f"Text length: {len(transcribed_text)} characters")
    print(f"Tokens used: {summary['total_tokens']:,}")
    print(f"  - Input: {summary['total_input_tokens']:,}")
    print(f"  - Output: {summary['total_output_tokens']:,}")
    print(f"Estimated cost: ${summary['total_cost']:.4f} USD")
    print()
    print(f"Validation: {'⚠️  NEEDS REVIEW' if validation_results['needs_review'] else '✓ PASSED'}")
    if validation_results['needs_review']:
        print(f"  Issues found: {len(validation_results['review_reasons'])}")
        for reason in validation_results['review_reasons']:
            print(f"    - {reason}")
    print()
    print("Files created:")
    print(f"  Transcription: {output_file}")
    print(f"  Summary: {summary_file}")
    
    # Show preview of transcribed text
    if transcribed_text and not transcribed_text.startswith('[ERROR'):
        print(f"\nText preview:")
        preview = transcribed_text[:300].replace('\n', ' ').strip()
        print(f"{preview}{'...' if len(transcribed_text) > 300 else ''}")
    elif transcribed_text.startswith('[ERROR'):
        print(f"\nError: {transcribed_text}")

if __name__ == "__main__":
    main()
