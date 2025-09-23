#!/usr/bin/env python3
"""
Hybrid OCR + AI Correction Script
Uses Tesseract OCR for initial text extraction, then OpenAI's API to correct
OCR errors, fix formatting, and improve readability while preserving historical context.
"""

import os
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import pytesseract
import openai
from openai import OpenAI

# OpenAI API pricing (as of 2024 - verify current rates)
PRICING = {
    'gpt-4o': {
        'input_tokens': 0.005 / 1000,     # $0.005 per 1K tokens  
        'output_tokens': 0.015 / 1000,    # $0.015 per 1K tokens
    },
    'gpt-4o-mini': {
        'input_tokens': 0.00015 / 1000,   # $0.00015 per 1K tokens
        'output_tokens': 0.0006 / 1000,   # $0.0006 per 1K tokens
    }
}

class TokenTracker:
    """Track token usage and calculate costs for OpenAI API calls."""
    
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.model_usage = {}
        self.ocr_processing_time = 0
        self.ai_correction_time = 0
    
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
    
    def add_timing(self, ocr_time: float, ai_time: float):
        """Add processing time tracking."""
        self.ocr_processing_time += ocr_time
        self.ai_correction_time += ai_time
    
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
            'ocr_time': self.ocr_processing_time,
            'ai_time': self.ai_correction_time,
            'total_time': self.ocr_processing_time + self.ai_correction_time,
            'model_breakdown': self.model_usage
        }

def extract_text_with_tesseract(image_path: str) -> Tuple[str, float]:
    """
    Extract text from image using Tesseract OCR.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        Tuple[str, float]: Extracted text and confidence score
    """
    try:
        start_time = time.time()
        
        # Open and process image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Use PSM 3 (Fully automatic page segmentation, but no OSD)
            custom_config = r'--oem 3 --psm 3'
            
            # Extract text and data
            text = pytesseract.image_to_string(img, config=custom_config)
            data = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Calculate confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            processing_time = time.time() - start_time
            
            return text.strip(), confidence, processing_time
            
    except Exception as e:
        print(f"Error processing {image_path} with Tesseract: {str(e)}")
        return "", 0.0, 0.0

def correct_text_with_ai(ocr_text: str, client: OpenAI, tracker: TokenTracker, model: str = "gpt-4o-mini") -> str:
    """
    Correct OCR text using OpenAI's API.
    
    Args:
        ocr_text (str): Raw OCR text to correct
        client (OpenAI): OpenAI client instance
        tracker (TokenTracker): Token usage tracker
        model (str): Model to use for text correction
        
    Returns:
        str: Corrected text
    """
    try:
        start_time = time.time()
        
        # Prepare the correction prompt
        prompt = f"""You are an expert in correcting OCR text from historical documents, particularly legal documents like deeds and court records from the 1800s. Please correct the following OCR text according to these guidelines:

CRITICAL REQUIREMENTS:
1. Process the ENTIRE document from beginning to end - do not stop early or truncate
2. Fix obvious OCR errors (0→O, 1→l, rn→m, etc.)
3. Add appropriate punctuation and capitalization where clearly needed
4. Fix spacing and line breaks for readability
5. Preserve original meaning and historical context exactly
6. If a word is unclear, make your best guess based on historical context
7. Do not add any text that wasn't in the original
8. Do not delete any substantive content unless it's clearly an OCR artifact
9. Preserve all names, dates, legal terms, and document structure
10. Maintain the formal legal language style of the era

OCR Text to Correct:
{ocr_text}

Return ONLY the corrected text with no additional commentary or explanations."""

        # Make API call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at correcting OCR text from historical legal documents. Focus on accuracy and preserving the original meaning while fixing obvious errors."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=4000,
            temperature=0.1  # Low temperature for consistent corrections
        )
        
        # Track token usage
        usage = response.usage
        tracker.add_usage(model, usage.prompt_tokens, usage.completion_tokens)
        
        processing_time = time.time() - start_time
        
        return response.choices[0].message.content.strip(), processing_time
        
    except Exception as e:
        print(f"Error correcting text with AI: {str(e)}")
        return ocr_text, 0.0  # Return original text if correction fails

def save_results(image_name: str, original_text: str, corrected_text: str, 
                confidence: float, output_dir: str) -> Tuple[str, str]:
    """
    Save both original OCR and corrected text to files.
    
    Args:
        image_name (str): Name of the source image
        original_text (str): Original OCR text
        corrected_text (str): AI-corrected text
        confidence (float): OCR confidence score
        output_dir (str): Directory to save results
        
    Returns:
        Tuple[str, str]: Paths to original and corrected text files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = Path(image_name).stem
    
    # Save original OCR text
    original_file = os.path.join(output_dir, f"{base_name}_original_ocr.txt")
    with open(original_file, 'w', encoding='utf-8') as f:
        f.write(f"Source Image: {image_name}\n")
        f.write(f"OCR Confidence: {confidence:.2f}%\n")
        f.write(f"Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("ORIGINAL OCR TEXT:\n")
        f.write("-" * 20 + "\n\n")
        f.write(original_text)
    
    # Save corrected text
    corrected_file = os.path.join(output_dir, f"{base_name}_corrected.txt")
    with open(corrected_file, 'w', encoding='utf-8') as f:
        f.write(f"Source Image: {image_name}\n")
        f.write(f"Original OCR Confidence: {confidence:.2f}%\n")
        f.write(f"Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("AI-CORRECTED TEXT:\n")
        f.write("-" * 18 + "\n\n")
        f.write(corrected_text)
    
    return original_file, corrected_file

def save_comparison_report(results: List[Dict], output_dir: str) -> None:
    """
    Save a comparison report showing original vs corrected text.
    
    Args:
        results (List[Dict]): Processing results for all images
        output_dir (str): Directory to save the report
    """
    report_file = os.path.join(output_dir, "correction_comparison.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("OCR + AI Correction Comparison Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Images Processed: {len(results)}\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"[{i}] {result['image']}\n")
            f.write(f"OCR Confidence: {result['confidence']:.2f}%\n")
            f.write(f"Original Length: {len(result['original'])} characters\n")
            f.write(f"Corrected Length: {len(result['corrected'])} characters\n")
            f.write("\nOriginal OCR Preview:\n")
            original_preview = result['original'][:200].replace('\n', ' ')
            f.write(f"{original_preview}{'...' if len(result['original']) > 200 else ''}\n\n")
            f.write("Corrected Text Preview:\n")
            corrected_preview = result['corrected'][:200].replace('\n', ' ')
            f.write(f"{corrected_preview}{'...' if len(result['corrected']) > 200 else ''}\n\n")
            f.write("-" * 50 + "\n\n")

def save_usage_summary(tracker: TokenTracker, output_dir: str, results: List[Dict]) -> None:
    """
    Save detailed usage and cost summary.
    
    Args:
        tracker (TokenTracker): Token usage tracker
        output_dir (str): Directory to save the summary
        results (List[Dict]): Processing results for all images
    """
    summary = tracker.get_summary()
    
    # Save as JSON
    json_file = os.path.join(output_dir, "processing_summary.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'processing_summary': summary,
            'images_processed': [r['image'] for r in results],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'estimated_cost_usd': summary['total_cost'],
            'avg_ocr_confidence': sum(r['confidence'] for r in results) / len(results) if results else 0
        }, f, indent=2)
    
    # Save as readable text
    text_file = os.path.join(output_dir, "processing_summary.txt")
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write("Hybrid OCR + AI Correction - Processing Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Images Processed: {len(results)}\n")
        f.write(f"Total API Requests: {summary['total_requests']}\n\n")
        
        f.write("Processing Time:\n")
        f.write("-" * 16 + "\n")
        f.write(f"OCR Processing: {summary['ocr_time']:.1f} seconds\n")
        f.write(f"AI Correction: {summary['ai_time']:.1f} seconds\n")
        f.write(f"Total Time: {summary['total_time']:.1f} seconds\n\n")
        
        f.write("Token Usage:\n")
        f.write("-" * 12 + "\n")
        f.write(f"Input Tokens:  {summary['total_input_tokens']:,}\n")
        f.write(f"Output Tokens: {summary['total_output_tokens']:,}\n")
        f.write(f"Total Tokens:  {summary['total_tokens']:,}\n\n")
        
        f.write(f"Estimated Cost: ${summary['total_cost']:.4f} USD\n\n")
        
        if results:
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            f.write(f"Average OCR Confidence: {avg_confidence:.2f}%\n\n")
        
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
    """Main function to process all images with hybrid OCR + AI correction."""
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Initialize clients and tracker
    client = OpenAI(api_key=api_key)
    tracker = TokenTracker()
    
    # Define paths
    base_dir = Path(__file__).parent
    images_dir = base_dir / "dupickens" / "dupickens_b-1" / "images"
    output_dir = base_dir / "ocr_ai_results"
    
    print("Starting Hybrid OCR + AI Correction processing...")
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
    print(f"\nProcessing images with Tesseract OCR + AI correction...")
    start_time = time.time()
    results = []
    
    for i, img_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {img_file.name}")
        
        # Step 1: Extract text with Tesseract
        print("  Step 1: Extracting text with Tesseract OCR...")
        original_text, confidence, ocr_time = extract_text_with_tesseract(str(img_file))
        
        if not original_text:
            print("  Warning: No text extracted by Tesseract")
            continue
        
        print(f"  OCR completed - Confidence: {confidence:.2f}%, Length: {len(original_text)} chars")
        
        # Step 2: Correct text with AI
        print("  Step 2: Correcting text with OpenAI...")
        corrected_text, ai_time = correct_text_with_ai(original_text, client, tracker)
        
        # Track timing
        tracker.add_timing(ocr_time, ai_time)
        
        # Step 3: Save results
        original_file, corrected_file = save_results(
            img_file.name, original_text, corrected_text, confidence, str(output_dir)
        )
        
        # Store results for comparison
        results.append({
            'image': img_file.name,
            'original': original_text,
            'corrected': corrected_text,
            'confidence': confidence,
            'original_file': original_file,
            'corrected_file': corrected_file
        })
        
        print(f"  Corrected length: {len(corrected_text)} characters")
        print(f"  Files saved: {Path(original_file).name}, {Path(corrected_file).name}")
        
        # Show preview of corrections
        if corrected_text:
            preview = corrected_text[:150].replace('\n', ' ').strip()
            print(f"  Corrected preview: {preview}{'...' if len(corrected_text) > 150 else ''}")
        
        # Small delay to respect API rate limits
        if i < len(image_files):
            time.sleep(0.5)
    
    # Calculate and display final summary
    end_time = time.time()
    total_processing_time = end_time - start_time
    summary = tracker.get_summary()
    
    print(f"\n" + "=" * 70)
    print("Hybrid OCR + AI Correction Processing Complete!")
    print("=" * 70)
    print(f"Images processed: {len(results)}")
    print(f"Total processing time: {total_processing_time:.1f} seconds")
    print(f"  - OCR time: {summary['ocr_time']:.1f} seconds")
    print(f"  - AI correction time: {summary['ai_time']:.1f} seconds")
    print(f"Total API requests: {summary['total_requests']}")
    print(f"Total tokens used: {summary['total_tokens']:,}")
    print(f"  - Input tokens: {summary['total_input_tokens']:,}")
    print(f"  - Output tokens: {summary['total_output_tokens']:,}")
    print(f"Estimated cost: ${summary['total_cost']:.4f} USD")
    
    if results:
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"Average OCR confidence: {avg_confidence:.2f}%")
    
    print(f"\nResults saved in: {output_dir}")
    
    # Save summary reports
    save_usage_summary(tracker, str(output_dir), results)
    save_comparison_report(results, str(output_dir))
    
    print("Summary files created:")
    print("  - processing_summary.txt (detailed usage and costs)")
    print("  - processing_summary.json (machine-readable data)")
    print("  - correction_comparison.txt (before/after comparison)")

if __name__ == "__main__":
    main()