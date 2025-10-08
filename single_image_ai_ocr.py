#!/usr/bin/env python3
"""
Single Image AI Vision OCR Script
Uses OpenAI's GPT-4 Vision to extract text from a single specified image while preserving layout,
structure, and handling multiple languages. Provides accurate transcription
without interpretation or translation.

Usage:
    # Using virtual environment (recommended):
    /path/to/.venv/bin/python single_image_ai_ocr.py path/to/image.jpg
    
    # Or activate virtual environment first:
    source .venv/bin/activate
    python single_image_ai_ocr.py path/to/image.jpg
"""

import os
import sys
import base64
import json
import time
from pathlib import Path
from typing import Dict, Tuple
from PIL import Image
import openai
from openai import OpenAI

# OpenAI API pricing (as of 2024 - verify current rates)
PRICING = {
    'gpt-4-vision-preview': {
        'input_tokens': 0.01 / 1000,      # $0.01 per 1K tokens
        'output_tokens': 0.03 / 1000,     # $0.03 per 1K tokens
    },
    'gpt-4o': {
        'input_tokens': 0.005 / 1000,     # $0.005 per 1K tokens  
        'output_tokens': 0.015 / 1000,    # $0.015 per 1K tokens
    }
}

class TokenTracker:
    """Track token usage and calculate costs for OpenAI API calls."""
    
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

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image to base64 string for OpenAI API.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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

def extract_text_with_ai_vision(image_path: str, client: OpenAI, tracker: TokenTracker, model: str = "gpt-4o") -> str:
    """
    Extract text from image using OpenAI's Vision API.
    
    Args:
        image_path (str): Path to the image file
        client (OpenAI): OpenAI client instance
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
        base64_image = encode_image_to_base64(optimized_path)
        
        # Prepare the prompt for accurate transcription
        prompt = """Extract ALL text from this image exactly as it appears. Follow these strict guidelines:

1. Transcribe every piece of visible text, preserving the original layout and structure
2. Maintain line breaks, paragraph breaks, and spatial relationships
3. Do not add any commentary, interpretation, or explanations
4. If text is unclear or partially obscured, transcribe what you can see - use [?] only for completely illegible characters
5. Preserve original spelling, punctuation, and formatting exactly as written
6. Include headers, titles, dates, numbers, and all visible text elements
7. If text is in a non-Latin script (Arabic, Chinese, Tamil, etc.), transcribe it exactly as written
8. Do not translate anything - only transcribe in the original language
9. Preserve the spatial layout - use appropriate spacing and line breaks to match the original
10. If there are multiple columns or sections, maintain their structure

Return ONLY the transcribed text with no additional commentary."""

        # Make API call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4000,
            temperature=0  # Use 0 for consistent, deterministic transcription
        )
        
        # Track token usage
        usage = response.usage
        tracker.add_usage(model, usage.prompt_tokens, usage.completion_tokens)
        
        # Track processing time
        processing_time = time.time() - start_time
        tracker.processing_time = processing_time
        
        # Clean up optimized image if it was created
        if optimized_path != image_path and os.path.exists(optimized_path):
            os.remove(optimized_path)
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        # Clean up optimized image on error
        if 'optimized_path' in locals() and optimized_path != image_path and os.path.exists(optimized_path):
            os.remove(optimized_path)
        return f"[ERROR: Could not process image - {str(e)}]"

def save_transcription(image_path: str, transcribed_text: str, output_dir: str = None) -> str:
    """
    Save transcribed text to a file.
    
    Args:
        image_path (str): Path to the source image
        transcribed_text (str): Extracted text
        output_dir (str): Directory to save the transcription (optional)
        
    Returns:
        str: Path to saved file
    """
    image_path_obj = Path(image_path)
    
    # Determine output directory
    if output_dir is None:
        # Use ocr_results_ai_vision structure like other scripts
        # Extract folder name from path (e.g., dupickens_b-1 from dupickens/dupickens_b-1/images/...)
        folder_name = None
        for part in image_path_obj.parts:
            if part.startswith('dupickens_'):
                folder_name = part
                break
        
        if folder_name:
            output_dir = Path("ocr_results_ai_vision") / folder_name
        else:
            # Fallback to old behavior if folder pattern not found
            output_dir = image_path_obj.parent / "transcriptions"
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
        f.write("=" * 60 + "\n\n")
        f.write(transcribed_text)
    
    return str(output_file)

def save_usage_summary(tracker: TokenTracker, output_file: str, image_path: str) -> None:
    """
    Save detailed usage and cost summary.
    
    Args:
        tracker (TokenTracker): Token usage tracker
        output_file (str): Path to save the summary
        image_path (str): Path to the processed image
    """
    summary = tracker.get_summary()
    
    # Save summary as text
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Single Image AI Vision OCR - Processing Summary\n")
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

def print_usage():
    """Print usage instructions."""
    print("Single Image AI Vision OCR")
    print("=" * 30)
    print()
    print("Usage:")
    print("  # Using virtual environment (recommended):")
    print("  /path/to/.venv/bin/python single_image_ai_ocr.py <image_path> [output_dir]")
    print()
    print("  # Or activate virtual environment first:")
    print("  source .venv/bin/activate")
    print("  python single_image_ai_ocr.py <image_path> [output_dir]")
    print()
    print("Arguments:")
    print("  image_path   Path to the image file to process")
    print("  output_dir   Optional output directory (default: ocr_results_ai_vision/[folder_name])")
    print()
    print("Examples:")
    print("  # From this project directory:")
    print("  ./.venv/bin/python single_image_ai_ocr.py dupickens/dupickens_b-1/images/dupickens_b-1_217.jpg")
    print("  ./.venv/bin/python single_image_ai_ocr.py image.jpg /custom/output")
    print()
    print("  # With activated virtual environment:")
    print("  source .venv/bin/activate")
    print("  python single_image_ai_ocr.py dupickens/dupickens_b-1/images/dupickens_b-1_217.jpg")
    print()
    print("Supported formats: .jpg, .jpeg, .png, .tiff, .bmp, .webp")

def main():
    """Main function to process a single image with AI vision."""
    
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
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp'}
    if not Path(image_path).suffix.lower() in image_extensions:
        print(f"Error: Unsupported file format. Supported formats: {', '.join(image_extensions)}")
        sys.exit(1)
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Initialize OpenAI client and tracker
    client = OpenAI(api_key=api_key)
    tracker = TokenTracker()
    
    print("Single Image AI Vision OCR")
    print("=" * 30)
    print(f"Processing: {Path(image_path).name}")
    print(f"Full path: {image_path}")
    if output_dir:
        print(f"Output directory: {output_dir}")
    print()
    
    # Process the image
    start_time = time.time()
    
    print("Extracting text with AI Vision...")
    transcribed_text = extract_text_with_ai_vision(image_path, client, tracker, model="gpt-4o")
    
    # Save transcription
    print("Saving transcription...")
    output_file = save_transcription(image_path, transcribed_text, output_dir)
    
    # Save usage summary in dedicated summaries subfolder
    summary_dir = Path(output_file).parent / "summaries"
    summary_dir.mkdir(exist_ok=True)
    summary_file = summary_dir / f"{Path(image_path).stem}_summary.txt"
    save_usage_summary(tracker, str(summary_file), image_path)
    
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