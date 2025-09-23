#!/usr/bin/env python3
"""
AI Vision-Based OCR Script
Uses OpenAI's GPT-4 Vision to extract text from images while preserving layout,
structure, and handling multiple languages. Provides accurate transcription
without interpretation or translation.
"""

import os
import base64
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
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

def save_transcription(image_name: str, transcribed_text: str, output_dir: str) -> str:
    """
    Save transcribed text to a file.
    
    Args:
        image_name (str): Name of the source image
        transcribed_text (str): Extracted text
        output_dir (str): Directory to save the transcription
        
    Returns:
        str: Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = Path(image_name).stem
    output_file = os.path.join(output_dir, f"{base_name}_transcription.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Source Image: {image_name}\n")
        f.write(f"Transcription Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(transcribed_text)
    
    return output_file

def save_usage_summary(tracker: TokenTracker, output_dir: str, image_files: List[str]) -> None:
    """
    Save detailed usage and cost summary.
    
    Args:
        tracker (TokenTracker): Token usage tracker
        output_dir (str): Directory to save the summary
        image_files (List[str]): List of processed image files
    """
    summary = tracker.get_summary()
    
    # Save as JSON
    json_file = os.path.join(output_dir, "usage_summary.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'processing_summary': summary,
            'images_processed': [str(Path(img).name) for img in image_files],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'estimated_cost_usd': summary['total_cost']
        }, f, indent=2)
    
    # Save as readable text
    text_file = os.path.join(output_dir, "usage_summary.txt")
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write("AI Vision OCR - Usage and Cost Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Images Processed: {len(image_files)}\n")
        f.write(f"Total API Requests: {summary['total_requests']}\n\n")
        
        f.write("Token Usage:\n")
        f.write("-" * 15 + "\n")
        f.write(f"Input Tokens:  {summary['total_input_tokens']:,}\n")
        f.write(f"Output Tokens: {summary['total_output_tokens']:,}\n")
        f.write(f"Total Tokens:  {summary['total_tokens']:,}\n\n")
        
        f.write(f"Estimated Cost: ${summary['total_cost']:.4f} USD\n\n")
        
        if summary['model_breakdown']:
            f.write("Model Breakdown:\n")
            f.write("-" * 16 + "\n")
            for model, usage in summary['model_breakdown'].items():
                model_cost = 0
                if model in PRICING:
                    model_cost = (
                        usage['input'] * PRICING[model]['input_tokens'] +
                        usage['output'] * PRICING[model]['output_tokens']
                    )
                f.write(f"{model}:\n")
                f.write(f"  Requests: {usage['requests']}\n")
                f.write(f"  Input tokens: {usage['input']:,}\n")
                f.write(f"  Output tokens: {usage['output']:,}\n")
                f.write(f"  Cost: ${model_cost:.4f}\n\n")

def main():
    """Main function to process all images with AI vision."""
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    tracker = TokenTracker()
    
    # Define paths
    base_dir = Path(__file__).parent
    images_dir = base_dir / "dupickens" / "dupickens_b-1" / "images"
    output_dir = base_dir / "ocr_results_ai_vision"
    
    print("Starting AI Vision OCR processing...")
    print(f"Images directory: {images_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if images directory exists
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp'}
    image_files = [f for f in images_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print("No image files found in the directory.")
        return
    
    print(f"Found {len(image_files)} image files:")
    for img_file in image_files:
        print(f"  - {img_file.name}")
    
    # Process each image
    print(f"\nProcessing images with AI Vision...")
    start_time = time.time()
    
    for i, img_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {img_file.name}")
        
        # Extract text using AI vision
        transcribed_text = extract_text_with_ai_vision(
            str(img_file), client, tracker, model="gpt-4o"
        )
        
        # Save transcription
        output_file = save_transcription(img_file.name, transcribed_text, str(output_dir))
        
        print(f"  Saved to: {Path(output_file).name}")
        print(f"  Text length: {len(transcribed_text)} characters")
        
        # Show preview of transcribed text
        if transcribed_text and not transcribed_text.startswith('[ERROR'):
            preview = transcribed_text[:150].replace('\n', ' ').strip()
            print(f"  Preview: {preview}{'...' if len(transcribed_text) > 150 else ''}")
        
        # Small delay to respect API rate limits
        if i < len(image_files):
            time.sleep(1)
    
    # Calculate and display final summary
    end_time = time.time()
    processing_time = end_time - start_time
    summary = tracker.get_summary()
    
    print(f"\n" + "=" * 60)
    print("AI Vision OCR Processing Complete!")
    print("=" * 60)
    print(f"Images processed: {len(image_files)}")
    print(f"Processing time: {processing_time:.1f} seconds")
    print(f"Total API requests: {summary['total_requests']}")
    print(f"Total tokens used: {summary['total_tokens']:,}")
    print(f"  - Input tokens: {summary['total_input_tokens']:,}")
    print(f"  - Output tokens: {summary['total_output_tokens']:,}")
    print(f"Estimated cost: ${summary['total_cost']:.4f} USD")
    print(f"\nResults saved in: {output_dir}")
    
    # Save detailed usage summary
    save_usage_summary(tracker, str(output_dir), [str(f) for f in image_files])
    print(f"Usage summary saved to: usage_summary.txt and usage_summary.json")

if __name__ == "__main__":
    main()