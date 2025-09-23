#!/usr/bin/env python3
"""
Tesseract OCR Script for Processing Images
Processes all images in dupickens/dupickens_b-1/images/ directory using Tesseract OCR
with PSM 3, includes text cleaning and confidence scoring.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import pytesseract

def clean_text(text: str) -> str:
    """
    Clean and format extracted text.
    
    Args:
        text (str): Raw OCR text
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace and normalize line breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Replace multiple newlines with double newline
    text = re.sub(r'[ \t]+', ' ', text)      # Replace multiple spaces/tabs with single space
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # Strip leading/trailing whitespace from lines
    
    # Remove common OCR artifacts
    text = re.sub(r'[^\w\s\.,;:!?\-\(\)\[\]\"\'\/]', '', text)  # Keep only alphanumeric and common punctuation
    
    # Fix common OCR errors
    replacements = {
        r'\b0\b': 'O',      # Replace standalone 0 with O
        r'\bl\b': 'I',      # Replace standalone l with I
        r'\brn\b': 'm',     # Replace rn with m
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text.strip()

def calculate_confidence_score(data: Dict) -> float:
    """
    Calculate average confidence score from Tesseract data.
    
    Args:
        data (Dict): Tesseract data dictionary
        
    Returns:
        float: Average confidence score (0-100)
    """
    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
    return sum(confidences) / len(confidences) if confidences else 0.0

def process_image(image_path: str) -> Tuple[str, float, Dict]:
    """
    Process a single image with Tesseract OCR.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        Tuple[str, float, Dict]: Cleaned text, confidence score, and raw data
    """
    try:
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
            
            # Clean text and calculate confidence
            cleaned_text = clean_text(text)
            confidence = calculate_confidence_score(data)
            
            return cleaned_text, confidence, data
            
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return "", 0.0, {}

def save_results(results: Dict, output_dir: str) -> None:
    """
    Save OCR results to text files.
    
    Args:
        results (Dict): Dictionary containing OCR results for each image
        output_dir (str): Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual results
    for image_name, result in results.items():
        base_name = Path(image_name).stem
        
        # Save cleaned text
        text_file = os.path.join(output_dir, f"{base_name}_ocr.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(f"Image: {image_name}\n")
            f.write(f"Confidence Score: {result['confidence']:.2f}%\n")
            f.write(f"{'='*50}\n\n")
            f.write(result['text'])
        
        # Save detailed data as JSON
        json_file = os.path.join(output_dir, f"{base_name}_data.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'image': image_name,
                'confidence': result['confidence'],
                'text': result['text'],
                'raw_data': result['data']
            }, f, indent=2, ensure_ascii=False)
    
    # Save summary report
    summary_file = os.path.join(output_dir, "ocr_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("OCR Processing Summary\n")
        f.write("=" * 30 + "\n\n")
        
        total_images = len(results)
        avg_confidence = sum(r['confidence'] for r in results.values()) / total_images if total_images > 0 else 0
        
        f.write(f"Total Images Processed: {total_images}\n")
        f.write(f"Average Confidence Score: {avg_confidence:.2f}%\n\n")
        
        f.write("Individual Results:\n")
        f.write("-" * 20 + "\n")
        
        for image_name, result in results.items():
            f.write(f"{image_name}: {result['confidence']:.2f}% confidence\n")
            f.write(f"  Text length: {len(result['text'])} characters\n")
            f.write(f"  Text preview: {result['text'][:100]}{'...' if len(result['text']) > 100 else ''}\n\n")

def main():
    """Main function to process all images in the target directory."""
    
    # Define paths
    base_dir = Path(__file__).parent
    images_dir = base_dir / "dupickens" / "dupickens_b-1" / "images"
    output_dir = base_dir / "ocr_results_tesseract"
    
    print("Starting OCR processing...")
    print(f"Images directory: {images_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if images directory exists
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
    image_files = [f for f in images_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print("No image files found in the directory.")
        return
    
    print(f"Found {len(image_files)} image files:")
    for img_file in image_files:
        print(f"  - {img_file.name}")
    
    # Process each image
    results = {}
    for img_file in image_files:
        print(f"\nProcessing: {img_file.name}")
        text, confidence, data = process_image(str(img_file))
        
        results[img_file.name] = {
            'text': text,
            'confidence': confidence,
            'data': data
        }
        
        print(f"  Confidence: {confidence:.2f}%")
        print(f"  Text length: {len(text)} characters")
        if text:
            preview = text[:100].replace('\n', ' ')
            print(f"  Preview: {preview}{'...' if len(text) > 100 else ''}")
    
    # Save results
    print(f"\nSaving results to: {output_dir}")
    save_results(results, str(output_dir))
    
    print("\nOCR processing completed!")
    print(f"Results saved in: {output_dir}")
    print("Files created:")
    print("  - Individual text files (*_ocr.txt)")
    print("  - Individual data files (*_data.json)")
    print("  - Summary report (ocr_summary.txt)")

if __name__ == "__main__":
    main()