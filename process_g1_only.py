#!/usr/bin/env python3
"""
Process Only dupickens_g-1 with AI Vision OCR
"""

import os
import time
from pathlib import Path
from typing import Dict, List
from ai_vision_ocr import (
    extract_text_with_ai_vision,
    save_transcription,
    TokenTracker,
    HallucinationDetector,
    save_usage_summary
)
from openai import OpenAI

def main():
    """Process only dupickens_g-1 folder."""
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Initialize
    client = OpenAI(api_key=api_key)
    tracker = TokenTracker()
    detector = HallucinationDetector()
    
    base_dir = Path(__file__).parent
    
    print("=" * 70)
    print("Processing dupickens_g-1 with AI Vision OCR")
    print("=" * 70)
    print()
    
    # Define paths
    folder_name = "dupickens_g-1"
    images_dir = base_dir / "dupickens" / folder_name / "images"
    output_dir = base_dir / "ocr_results_ai_vision" / folder_name
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp'}
    image_files = [f for f in images_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {images_dir}")
        return
    
    image_files = sorted(image_files)
    
    print(f"Found {len(image_files)} image files")
    print(f"Images directory: {images_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Ensure output directory exists
    os.makedirs(str(output_dir), exist_ok=True)
    
    # Process each image
    folder_results = []
    start_time = time.time()
    
    for i, img_file in enumerate(image_files, 1):
        # Check if already processed
        transcription_file = output_dir / f"{img_file.stem}_transcription.txt"
        if transcription_file.exists():
            print(f"[{i}/{len(image_files)}] Skipping {img_file.name} - already transcribed")
            continue
        
        print(f"[{i}/{len(image_files)}] Processing: {img_file.name}")
        
        # Extract text using AI vision
        transcribed_text, analysis_results = extract_text_with_ai_vision(
            str(img_file), client, tracker, detector, model="gpt-4o"
        )
        
        # If hallucinations detected, clean the text
        if analysis_results['needs_review']:
            print("  ⚠️ Quality issues detected:")
            for reason in analysis_results['review_reasons']:
                print(f"    - {reason}")
            print("  Cleaning text...")
            transcribed_text = detector.clean_hallucinations(transcribed_text)
        
        # Save transcription
        output_file = save_transcription(str(img_file), transcribed_text, str(output_dir), analysis_results)
        
        if output_file:
            print(f"  ✓ Saved: {Path(output_file).name}")
            print(f"  Text length: {len(transcribed_text)} characters")
            
            # Show preview
            if transcribed_text and not transcribed_text.startswith('[ERROR'):
                preview = transcribed_text[:150].replace('\n', ' ').strip()
                print(f"  Preview: {preview}{'...' if len(transcribed_text) > 150 else ''}")
            
            folder_results.append({
                'image': img_file.name,
                'text': transcribed_text,
                'output_file': str(output_file),
                'analysis': analysis_results
            })
        else:
            print("  ✗ Error: Could not save transcription")
        
        # Small delay to respect API rate limits
        if i < len(image_files):
            time.sleep(1)
        
        print()
    
    # Calculate totals
    processing_time = time.time() - start_time
    summary = tracker.get_summary()
    
    print("=" * 70)
    print("Processing Complete!")
    print("=" * 70)
    print(f"Folder: {folder_name}")
    print(f"Images processed: {len(folder_results)}")
    print(f"Processing time: {processing_time:.1f} seconds")
    print(f"Total API requests: {summary['total_requests']}")
    print(f"Total tokens used: {summary['total_tokens']:,}")
    print(f"  - Input tokens: {summary['total_input_tokens']:,}")
    print(f"  - Output tokens: {summary['total_output_tokens']:,}")
    print(f"Estimated cost: ${summary['total_cost']:.4f} USD")
    print()
    
    # Save usage summary
    all_image_files = [result['image'] for result in folder_results]
    save_usage_summary(tracker, str(output_dir.parent), all_image_files)
    print(f"Results saved in: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
