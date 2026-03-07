#!/usr/bin/env python3
"""
TIFF to JPG Conversion Script
Converts TIFF images to JPG format with progress tracking.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple
import time

try:
    from PIL import Image
except ImportError as e:
    print(f"Error: Required package not installed. {e}")
    print("Please install required packages:")
    print("pip install Pillow")
    sys.exit(1)


def convert_tif_to_jpg(input_path: Path, output_path: Path, quality: int = 95) -> Tuple[bool, str]:
    """
    Convert a single TIFF file to JPG format.
    
    Args:
        input_path: Path to input TIFF file
        output_path: Path to output JPG file
        quality: JPG quality (1-100, default 95)
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Open the TIFF image
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (TIFF might be in other modes)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Convert RGBA/LA to RGB by creating white background
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as JPG
            img.save(output_path, 'JPEG', quality=quality, optimize=True)
            
        return True, f"Successfully converted to {output_path.name}"
        
    except Exception as e:
        return False, f"Error converting {input_path.name}: {str(e)}"


def main():
    """Main conversion function."""
    
    # Define paths
    base_dir = Path("/Users/eahorton/Downloads/deeds_unbound")
    input_dir = base_dir / "dupickens" / "dupickens_f-1" / "images_tif" / "TIF_300DPI"
    output_dir = base_dir / "dupickens" / "dupickens_f-1" / "images"
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Find all TIF files
    tif_files = sorted(input_dir.glob("*.tif"))
    
    if not tif_files:
        print(f"No TIF files found in {input_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(tif_files)} TIF files to convert")
    print("=" * 60)
    
    # Conversion statistics
    successful = 0
    failed = 0
    errors = []
    start_time = time.time()
    
    # Convert each file
    for i, tif_path in enumerate(tif_files, 1):
        # Create output filename (replace .tif with .jpg)
        jpg_filename = tif_path.stem + ".jpg"
        jpg_path = output_dir / jpg_filename
        
        # Progress indicator
        print(f"[{i}/{len(tif_files)}] Converting {tif_path.name}...", end=" ")
        
        # Convert the file
        success, message = convert_tif_to_jpg(tif_path, jpg_path)
        
        if success:
            successful += 1
            print("✓")
        else:
            failed += 1
            errors.append(message)
            print("✗")
            print(f"    {message}")
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {len(tif_files)}")
    print(f"Successfully converted: {successful}")
    print(f"Failed: {failed}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Average time per file: {elapsed_time/len(tif_files):.2f} seconds")
    
    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"  - {error}")
    
    print(f"\nJPG files saved to: {output_dir}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
