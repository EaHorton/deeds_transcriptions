#!/usr/bin/env python3
"""
TIFF to PDF Conversion Script
Converts TIFF images from images_tif folders to PDF files and saves them 
in corresponding images_pdf folders.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import time

try:
    from PIL import Image
except ImportError as e:
    print(f"Error: Required package not installed. {e}")
    print("Please install required packages:")
    print("pip install Pillow")
    sys.exit(1)

class TIFFConverter:
    """Handle TIFF to PDF conversion with progress tracking and error handling."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.conversion_stats = {
            'folders_processed': 0,
            'tiffs_found': 0,
            'tiffs_converted': 0,
            'errors': [],
            'skipped': [],
            'processing_time': 0
        }
    
    def find_dupickens_folders(self) -> List[Path]:
        """
        Find all dupickens subfolders that contain images_tif directories.
        
        Returns:
            List[Path]: List of dupickens subfolder paths
        """
        dupickens_folders = []
        dupickens_main = self.base_dir / "dupickens"
        
        if not dupickens_main.exists():
            print(f"Error: dupickens folder not found at {dupickens_main}")
            return []
        
        # Find all subfolders that contain images_tif directories
        for item in dupickens_main.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                images_tif_dir = item / "images_tif"
                if images_tif_dir.exists():
                    dupickens_folders.append(item)
        
        return sorted(dupickens_folders)
    
    def get_tif_files(self, images_tif_dir: Path) -> List[Path]:
        """
        Get all TIFF files from an images_tif directory (including subdirectories).
        
        Args:
            images_tif_dir (Path): Path to images_tif directory
            
        Returns:
            List[Path]: List of TIFF file paths
        """
        tif_files = []
        if images_tif_dir.exists():
            # Search in the directory and all subdirectories
            for root, dirs, files in os.walk(images_tif_dir):
                root_path = Path(root)
                for file in files:
                    if file.lower().endswith(('.tif', '.tiff')) and not file.startswith('.'):
                        tif_files.append(root_path / file)
        return sorted(tif_files)
    
    def convert_tif_to_pdf(self, tif_path: Path, output_dir: Path) -> Tuple[bool, str, str]:
        """
        Convert a TIFF file to PDF.
        
        Args:
            tif_path (Path): Path to the TIFF file
            output_dir (Path): Directory to save PDF file
            
        Returns:
            Tuple[bool, str, str]: (success, created filename, error message)
        """
        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Open TIFF image
            print(f"    Converting {tif_path.name}...")
            image = Image.open(tif_path)
            
            # Convert to RGB if necessary (PDF requires RGB mode)
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            elif image.mode == 'L':
                # Keep grayscale images as-is
                pass
            
            # Create output filename
            base_name = tif_path.stem
            output_filename = f"{base_name}.pdf"
            output_path = output_dir / output_filename
            
            # Check for multi-page TIFF
            images = []
            try:
                # Try to handle multi-page TIFF
                for i in range(image.n_frames):
                    image.seek(i)
                    frame = image.copy()
                    if frame.mode not in ('RGB', 'L'):
                        frame = frame.convert('RGB')
                    images.append(frame)
            except AttributeError:
                # Single page TIFF
                images = [image]
            
            # Save as PDF
            if len(images) == 1:
                images[0].save(
                    str(output_path),
                    'PDF',
                    resolution=300.0,
                    save_all=False
                )
            else:
                # Multi-page TIFF to multi-page PDF
                images[0].save(
                    str(output_path),
                    'PDF',
                    resolution=300.0,
                    save_all=True,
                    append_images=images[1:]
                )
            
            # Close images
            for img in images:
                img.close()
            
            return True, output_filename, ""
            
        except Exception as e:
            error_msg = f"Error converting {tif_path.name}: {str(e)}"
            return False, "", error_msg
    
    def check_existing_pdf(self, tif_path: Path, output_dir: Path) -> bool:
        """
        Check if PDF file already exists for a TIFF.
        
        Args:
            tif_path (Path): Path to the TIFF file
            output_dir (Path): Output directory to check
            
        Returns:
            bool: True if corresponding PDF file exists
        """
        base_name = tif_path.stem
        pdf_file = output_dir / f"{base_name}.pdf"
        return pdf_file.exists()
    
    def process_folder(self, folder_path: Path, skip_existing: bool = True) -> Dict:
        """
        Process a single dupickens subfolder.
        
        Args:
            folder_path (Path): Path to the dupickens subfolder
            skip_existing (bool): Skip conversion if PDF already exists
            
        Returns:
            Dict: Processing results for this folder
        """
        folder_stats = {
            'folder_name': folder_path.name,
            'tiffs_found': 0,
            'tiffs_converted': 0,
            'tiffs_skipped': 0,
            'errors': []
        }
        
        images_tif_dir = folder_path / "images_tif"
        images_pdf_dir = folder_path / "images_pdf"
        
        # Get all TIFF files
        tif_files = self.get_tif_files(images_tif_dir)
        folder_stats['tiffs_found'] = len(tif_files)
        
        if not tif_files:
            print(f"  No TIFF files found in {images_tif_dir}")
            return folder_stats
        
        print(f"  Found {len(tif_files)} TIFF files")
        
        # Process each TIFF
        for tif_file in tif_files:
            # Check if PDF already exists
            if skip_existing and self.check_existing_pdf(tif_file, images_pdf_dir):
                print(f"    Skipping {tif_file.name} (PDF already exists)")
                folder_stats['tiffs_skipped'] += 1
                self.conversion_stats['skipped'].append(str(tif_file))
                continue
            
            # Convert TIFF to PDF
            success, output_file, error = self.convert_tif_to_pdf(tif_file, images_pdf_dir)
            
            if success:
                print(f"    ✓ Created {output_file}")
                folder_stats['tiffs_converted'] += 1
                self.conversion_stats['tiffs_converted'] += 1
            else:
                print(f"    ✗ {error}")
                folder_stats['errors'].append(error)
                self.conversion_stats['errors'].append(error)
        
        return folder_stats
    
    def process_all_folders(self, skip_existing: bool = True):
        """
        Process all dupickens folders with TIFF images.
        
        Args:
            skip_existing (bool): Skip conversion if PDF already exists
        """
        start_time = time.time()
        
        print("=" * 70)
        print("TIFF to PDF Conversion")
        print("=" * 70)
        print()
        
        # Find all folders with images_tif directories
        folders = self.find_dupickens_folders()
        
        if not folders:
            print("No folders with images_tif directories found.")
            return
        
        print(f"Found {len(folders)} folder(s) with images_tif directories:")
        for folder in folders:
            print(f"  - {folder.name}")
        print()
        
        # Process each folder
        all_folder_stats = []
        for folder in folders:
            print(f"\nProcessing {folder.name}...")
            print("-" * 70)
            folder_stats = self.process_folder(folder, skip_existing)
            all_folder_stats.append(folder_stats)
            self.conversion_stats['folders_processed'] += 1
            self.conversion_stats['tiffs_found'] += folder_stats['tiffs_found']
        
        # Calculate processing time
        self.conversion_stats['processing_time'] = time.time() - start_time
        
        # Print summary
        self.print_summary(all_folder_stats)
    
    def print_summary(self, folder_stats: List[Dict]):
        """
        Print a summary of the conversion process.
        
        Args:
            folder_stats (List[Dict]): Statistics for each folder
        """
        print("\n" + "=" * 70)
        print("CONVERSION SUMMARY")
        print("=" * 70)
        print()
        
        # Overall statistics
        print("Overall Statistics:")
        print(f"  Folders processed: {self.conversion_stats['folders_processed']}")
        print(f"  TIFF files found: {self.conversion_stats['tiffs_found']}")
        print(f"  PDFs created: {self.conversion_stats['tiffs_converted']}")
        print(f"  PDFs skipped (already exist): {len(self.conversion_stats['skipped'])}")
        print(f"  Errors: {len(self.conversion_stats['errors'])}")
        print(f"  Processing time: {self.conversion_stats['processing_time']:.2f} seconds")
        print()
        
        # Per-folder breakdown
        if len(folder_stats) > 1:
            print("Per-Folder Breakdown:")
            for stats in folder_stats:
                if stats['tiffs_found'] > 0:
                    print(f"\n  {stats['folder_name']}:")
                    print(f"    TIFF files found: {stats['tiffs_found']}")
                    print(f"    PDFs converted: {stats['tiffs_converted']}")
                    print(f"    PDFs skipped: {stats['tiffs_skipped']}")
                    if stats['errors']:
                        print(f"    Errors: {len(stats['errors'])}")
            print()
        
        # Print errors if any
        if self.conversion_stats['errors']:
            print("\nErrors encountered:")
            for error in self.conversion_stats['errors']:
                print(f"  - {error}")
            print()
        
        print("=" * 70)


def main():
    """Main function to run the TIFF to PDF conversion."""
    # Get the base directory (parent of this script)
    script_dir = Path(__file__).parent
    
    # Create converter instance
    converter = TIFFConverter(str(script_dir))
    
    # Process all folders (skip existing PDFs by default)
    # Set skip_existing=False to reconvert all files
    converter.process_all_folders(skip_existing=True)


if __name__ == "__main__":
    main()
