#!/usr/bin/env python3
"""
Reprocess Failed AI Vision OCR Images
Only processes images that don't have corresponding transcription files.
"""

import os
import time
from pathlib import Path
from ai_vision_ocr import (
    extract_text_with_ai_vision,
    save_transcription,
    TokenTracker,
    HallucinationDetector
)
from openai import OpenAI

# List of images that need to be reprocessed (files without transcription files)
FAILED_IMAGES_D1 = [
    "dupickens_d-1_300DPI_264.jpg",
    "dupickens_d-1_300DPI_307.jpg",
    "dupickens_d-1_300DPI_361.jpg",
    "dupickens_d-1_300DPI_333.jpg",
    "dupickens_d-1_608.jpg",
    "dupickens_d-1_642.jpg",
    "dupickens_d-1_300DPI_265.jpg",
    "dupickens_d-1_146.jpg",
    "dupickens_d-1_300DPI_536.jpg",
    "dupickens_d-1_300DPI_306.jpg",
    "dupickens_d-1_300DPI_360.jpg",
    "dupickens_d-1_570.jpg",
    "dupickens_d-1_145.jpg",
    "dupickens_d-1_571.jpg",
    "dupickens_d-1_636.jpg",
    "dupickens_d-1_137.jpg",
    "dupickens_d-1_607.jpg",
    "dupickens_d-1_580.jpg",
    "dupickens_d-1_300DPI_242.jpg",
    "dupickens_d-1_300DPI_472.jpg",
    "dupickens_d-1_551.jpg",
    "dupickens_d-1_634.jpg",
    "dupickens_d-1_534.jpg",
    "dupickens_d-1_635.jpg",
    "dupickens_d-1_630.jpg",
    "dupickens_d-1_331.jpg",
    "dupickens_d-1_535.jpg",
]

FAILED_IMAGES_E1 = [
    "dupickens_e-1_255.jpg",
    "dupickens_e-1_300DPI_483.jpg",
    "dupickens_e-1_300DPI_305.jpg",
    "dupickens_e-1_254.jpg",
    "dupickens_e-1_300DPI_482.jpg",
    "dupickens_e-1_300DPI_356.jpg",
    "dupickens_e-1_228.jpg",
    "dupickens_e-1_300DPI_306.jpg",
    "dupickens_e-1_246.jpg",
    "dupickens_e-1_300DPI_394.jpg",
    "dupickens_e-1_273.jpg",
    "dupickens_e-1_221.jpg",
    "dupickens_e-1_300DPI_413.jpg",
    "dupickens_e-1_300DPI_393.jpg",
    "dupickens_e-1_275.jpg",
    "dupickens_e-1_300DPI_414.jpg",
    "dupickens_e-1_245.jpg",
    "dupickens_e-1_274.jpg",
]

def main():
    """Process only the failed images."""
    
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
    base_output_dir = base_dir / "ocr_results_ai_vision"
    
    print("=" * 70)
    print("Reprocessing Failed Images")
    print("=" * 70)
    print()
    
    # Process dupickens_d-1
    if FAILED_IMAGES_D1:
        print(f"Processing dupickens_d-1: {len(FAILED_IMAGES_D1)} images")
        print("-" * 70)
        
        images_dir = base_dir / "dupickens" / "dupickens_d-1" / "images"
        output_dir = base_output_dir / "dupickens_d-1"
        
        for i, img_name in enumerate(FAILED_IMAGES_D1, 1):
            img_path = images_dir / img_name
            
            if not img_path.exists():
                print(f"  [{i}/{len(FAILED_IMAGES_D1)}] Skipping {img_name} - file not found")
                continue
            
            # Check if transcription already exists
            transcription_file = output_dir / f"{img_path.stem}_transcription.txt"
            if transcription_file.exists():
                print(f"  [{i}/{len(FAILED_IMAGES_D1)}] Skipping {img_name} - already transcribed")
                continue
            
            print(f"  [{i}/{len(FAILED_IMAGES_D1)}] Processing: {img_name}")
            
            # Extract text
            transcribed_text, analysis_results = extract_text_with_ai_vision(
                str(img_path), client, tracker, detector, model="gpt-4o"
            )
            
            # If hallucinations detected, clean the text
            if analysis_results['needs_review']:
                print("    ⚠️ Quality issues detected:")
                for reason in analysis_results['review_reasons']:
                    print(f"      - {reason}")
                print("    Cleaning text...")
                transcribed_text = detector.clean_hallucinations(transcribed_text)
            
            # Save transcription
            output_file = save_transcription(str(img_path), transcribed_text, str(output_dir), analysis_results)
            
            if output_file:
                print(f"    ✓ Saved: {Path(output_file).name}")
                print(f"    Text length: {len(transcribed_text)} characters")
            else:
                print("    ✗ Error: Could not save transcription")
            
            # Small delay to respect API rate limits
            if i < len(FAILED_IMAGES_D1):
                time.sleep(1)
        
        print()
    
    # Process dupickens_e-1
    if FAILED_IMAGES_E1:
        print(f"Processing dupickens_e-1: {len(FAILED_IMAGES_E1)} images")
        print("-" * 70)
        
        images_dir = base_dir / "dupickens" / "dupickens_e-1" / "images"
        output_dir = base_output_dir / "dupickens_e-1"
        
        for i, img_name in enumerate(FAILED_IMAGES_E1, 1):
            img_path = images_dir / img_name
            
            if not img_path.exists():
                print(f"  [{i}/{len(FAILED_IMAGES_E1)}] Skipping {img_name} - file not found")
                continue
            
            # Check if transcription already exists
            transcription_file = output_dir / f"{img_path.stem}_transcription.txt"
            if transcription_file.exists():
                print(f"  [{i}/{len(FAILED_IMAGES_E1)}] Skipping {img_name} - already transcribed")
                continue
            
            print(f"  [{i}/{len(FAILED_IMAGES_E1)}] Processing: {img_name}")
            
            # Extract text
            transcribed_text, analysis_results = extract_text_with_ai_vision(
                str(img_path), client, tracker, detector, model="gpt-4o"
            )
            
            # If hallucinations detected, clean the text
            if analysis_results['needs_review']:
                print("    ⚠️ Quality issues detected:")
                for reason in analysis_results['review_reasons']:
                    print(f"      - {reason}")
                print("    Cleaning text...")
                transcribed_text = detector.clean_hallucinations(transcribed_text)
            
            # Save transcription
            output_file = save_transcription(str(img_path), transcribed_text, str(output_dir), analysis_results)
            
            if output_file:
                print(f"    ✓ Saved: {Path(output_file).name}")
                print(f"    Text length: {len(transcribed_text)} characters")
            else:
                print("    ✗ Error: Could not save transcription")
            
            # Small delay to respect API rate limits
            if i < len(FAILED_IMAGES_E1):
                time.sleep(1)
        
        print()
    
    # Print summary
    summary = tracker.get_summary()
    print("=" * 70)
    print("Processing Complete!")
    print("=" * 70)
    print(f"Total images targeted: {len(FAILED_IMAGES_D1) + len(FAILED_IMAGES_E1)}")
    print(f"Total API requests: {summary['total_requests']}")
    print(f"Total tokens used: {summary['total_tokens']:,}")
    print(f"  - Input tokens: {summary['total_input_tokens']:,}")
    print(f"  - Output tokens: {summary['total_output_tokens']:,}")
    print(f"Estimated cost: ${summary['total_cost']:.4f} USD")
    print()

if __name__ == "__main__":
    main()
