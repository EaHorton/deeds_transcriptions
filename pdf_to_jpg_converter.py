#!/usr/bin/env python3
"""
PDF to JPG Conversion Script
Processes all subfolders in the dupickens directory, converts PDF files from 
images_pdf folders to high-quality JPG images, and saves them in corresponding images folders.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import time

try:
    from pdf2image import convert_from_path
    from PIL import Image
except ImportError as e:
    print(f"Error: Required packages not installed. {e}")
    print("Please install required packages:")
    print("pip install pdf2image Pillow")
    print("\nAdditional system requirements:")
    print("- macOS: brew install poppler")
    print("- Ubuntu/Debian: sudo apt-get install poppler-utils")
    print("- Windows: Download poppler and add to PATH")
    sys.exit(1)

class PDFConverter:
    """Handle PDF to JPG conversion with progress tracking and error handling."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.conversion_stats = {
            'folders_processed': 0,
            'pdfs_found': 0,
            'pdfs_converted': 0,
            'pages_converted': 0,
            'errors': [],
            'skipped': [],
            'processing_time': 0
        }
    
    def find_dupickens_folders(self) -> List[Path]:
        """
        Find all dupickens subfolders that contain images_pdf directories.
        
        Returns:
            List[Path]: List of dupickens subfolder paths
        """
        dupickens_folders = []
        dupickens_main = self.base_dir / "dupickens"
        
        if not dupickens_main.exists():
            print(f"Error: dupickens folder not found at {dupickens_main}")
            return []
        
        # Find all subfolders that contain images_pdf directories
        for item in dupickens_main.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                images_pdf_dir = item / "images_pdf"
                if images_pdf_dir.exists():
                    dupickens_folders.append(item)
        
        return sorted(dupickens_folders)
    
    def get_pdf_files(self, images_pdf_dir: Path) -> List[Path]:
        """
        Get all PDF files from an images_pdf directory.
        
        Args:
            images_pdf_dir (Path): Path to images_pdf directory
            
        Returns:
            List[Path]: List of PDF file paths
        """
        pdf_files = []
        if images_pdf_dir.exists():
            pdf_files = [f for f in images_pdf_dir.iterdir() 
                        if f.is_file() and f.suffix.lower() == '.pdf']
        return sorted(pdf_files)
    
    def convert_pdf_to_jpg(self, pdf_path: Path, output_dir: Path, 
                          dpi: int = 300, quality: int = 95) -> Tuple[bool, List[str], str]:
        """
        Convert a PDF file to JPG images.
        
        Args:
            pdf_path (Path): Path to the PDF file
            output_dir (Path): Directory to save JPG files
            dpi (int): DPI for conversion (higher = better quality)
            quality (int): JPEG quality (1-100)
            
        Returns:
            Tuple[bool, List[str], str]: (success, list of created files, error message)
        """
        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert PDF to images
            print(f"    Converting {pdf_path.name}...")
            images = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                fmt='JPEG',
                thread_count=2  # Use 2 threads for faster conversion
            )
            
            created_files = []
            base_name = pdf_path.stem  # filename without extension
            
            for page_num, image in enumerate(images, 1):
                # Create output filename
                if len(images) == 1:
                    # Single page PDF - use original name
                    output_filename = f"{base_name}.jpg"
                else:
                    # Multi-page PDF - add page number
                    output_filename = f"{base_name}_page_{page_num:03d}.jpg"
                
                output_path = output_dir / output_filename
                
                # Save as high-quality JPEG
                image.save(
                    str(output_path),
                    'JPEG',
                    quality=quality,
                    optimize=True,
                    dpi=(dpi, dpi)
                )
                
                created_files.append(output_filename)
                self.conversion_stats['pages_converted'] += 1
            
            return True, created_files, ""
            
        except Exception as e:
            error_msg = f"Error converting {pdf_path.name}: {str(e)}"
            return False, [], error_msg
    
    def check_existing_jpg(self, pdf_path: Path, output_dir: Path) -> bool:
        """
        Check if JPG files already exist for a PDF.
        
        Args:
            pdf_path (Path): Path to the PDF file
            output_dir (Path): Output directory to check
            
        Returns:
            bool: True if corresponding JPG files exist
        """
        base_name = pdf_path.stem
        
        # Check for single page version
        single_page_jpg = output_dir / f"{base_name}.jpg"
        if single_page_jpg.exists():
            return True
        
        # Check for multi-page versions
        multi_page_pattern = output_dir.glob(f"{base_name}_page_*.jpg")
        if any(multi_page_pattern):
            return True
        
        return False
    
    def process_folder(self, folder_path: Path, skip_existing: bool = True) -> Dict:
        """
        Process a single dupickens subfolder.
        
        Args:
            folder_path (Path): Path to the dupickens subfolder
            skip_existing (bool): Skip conversion if JPG already exists
            
        Returns:
            Dict: Processing results for this folder
        """
        folder_stats = {
            'folder_name': folder_path.name,
            'pdfs_found': 0,
            'pdfs_converted': 0,
            'pdfs_skipped': 0,
            'pages_created': 0,
            'errors': []
        }
        
        images_pdf_dir = folder_path / "images_pdf"
        images_dir = folder_path / "images"
        
        # Get all PDF files
        pdf_files = self.get_pdf_files(images_pdf_dir)
        folder_stats['pdfs_found'] = len(pdf_files)
        
        if not pdf_files:
            print(f"  No PDF files found in {images_pdf_dir}")
            return folder_stats
        
        print(f"  Found {len(pdf_files)} PDF files")
        
        # Process each PDF
        for pdf_file in pdf_files:
            # Check if JPG already exists
            if skip_existing and self.check_existing_jpg(pdf_file, images_dir):
                print(f"    Skipping {pdf_file.name} (JPG already exists)")
                folder_stats['pdfs_skipped'] += 1
                self.conversion_stats['skipped'].append(f"{folder_path.name}/{pdf_file.name}")
                continue
            
            # Convert PDF to JPG
            success, created_files, error_msg = self.convert_pdf_to_jpg(pdf_file, images_dir)
            
            if success:
                folder_stats['pdfs_converted'] += 1
                folder_stats['pages_created'] += len(created_files)
                print(f"    ✓ Created {len(created_files)} JPG file(s): {', '.join(created_files)}")
            else:
                folder_stats['errors'].append(error_msg)
                self.conversion_stats['errors'].append(f"{folder_path.name}: {error_msg}")
                print(f"    ✗ {error_msg}")
        
        return folder_stats
    
    def process_all_folders(self, skip_existing: bool = True, dpi: int = 300) -> None:
        """
        Process all dupickens subfolders.
        
        Args:
            skip_existing (bool): Skip conversion if JPG already exists
            dpi (int): DPI for conversion
        """
        start_time = time.time()
        
        print("PDF to JPG Conversion Script")
        print("=" * 40)
        print(f"Base directory: {self.base_dir}")
        print(f"DPI setting: {dpi}")
        print(f"Skip existing: {skip_existing}")
        print()
        
        # Find all dupickens folders
        folders = self.find_dupickens_folders()
        
        if not folders:
            print("No dupickens subfolders with images_pdf directories found.")
            return
        
        print(f"Found {len(folders)} dupickens subfolders to process:")
        for folder in folders:
            print(f"  - {folder.name}")
        print()
        
        # Process each folder
        all_folder_stats = []
        
        for i, folder in enumerate(folders, 1):
            print(f"[{i}/{len(folders)}] Processing {folder.name}...")
            folder_stats = self.process_folder(folder, skip_existing)
            all_folder_stats.append(folder_stats)
            
            # Update global stats
            self.conversion_stats['folders_processed'] += 1
            self.conversion_stats['pdfs_found'] += folder_stats['pdfs_found']
            self.conversion_stats['pdfs_converted'] += folder_stats['pdfs_converted']
            
            print()
        
        # Calculate final stats
        self.conversion_stats['processing_time'] = time.time() - start_time
        
        # Print summary
        self.print_summary(all_folder_stats)
    
    def print_summary(self, folder_stats: List[Dict]) -> None:
        """
        Print processing summary.
        
        Args:
            folder_stats (List[Dict]): Statistics for each folder
        """
        print("=" * 60)
        print("PROCESSING COMPLETE - SUMMARY")
        print("=" * 60)
        
        print(f"Total processing time: {self.conversion_stats['processing_time']:.1f} seconds")
        print(f"Folders processed: {self.conversion_stats['folders_processed']}")
        print(f"PDF files found: {self.conversion_stats['pdfs_found']}")
        print(f"PDF files converted: {self.conversion_stats['pdfs_converted']}")
        print(f"Total pages converted: {self.conversion_stats['pages_converted']}")
        print(f"Files skipped: {len(self.conversion_stats['skipped'])}")
        print(f"Errors encountered: {len(self.conversion_stats['errors'])}")
        
        # Detailed folder breakdown
        if folder_stats:
            print("\nDetailed Results by Folder:")
            print("-" * 30)
            for stats in folder_stats:
                print(f"{stats['folder_name']}:")
                print(f"  PDFs found: {stats['pdfs_found']}")
                print(f"  PDFs converted: {stats['pdfs_converted']}")
                print(f"  PDFs skipped: {stats['pdfs_skipped']}")
                print(f"  Pages created: {stats['pages_created']}")
                if stats['errors']:
                    print(f"  Errors: {len(stats['errors'])}")
        
        # Show errors if any
        if self.conversion_stats['errors']:
            print(f"\nErrors Encountered:")
            print("-" * 18)
            for error in self.conversion_stats['errors']:
                print(f"  {error}")
        
        # Show skipped files if any
        if self.conversion_stats['skipped']:
            print(f"\nSkipped Files (already exist):")
            print("-" * 30)
            for skipped in self.conversion_stats['skipped'][:10]:  # Show first 10
                print(f"  {skipped}")
            if len(self.conversion_stats['skipped']) > 10:
                print(f"  ... and {len(self.conversion_stats['skipped']) - 10} more")
        
        print(f"\nConversion complete! Check the 'images' folders in each dupickens subfolder for the JPG files.")

def main():
    """Main function to run the PDF conversion process."""
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Initialize converter
    converter = PDFConverter(script_dir)
    
    # Check if we have the required dependencies
    try:
        from pdf2image import convert_from_path
        from PIL import Image
    except ImportError:
        print("Error: Required packages not installed.")
        print("Please install:")
        print("pip install pdf2image Pillow")
        print("\nSystem requirements:")
        print("- macOS: brew install poppler")
        print("- Ubuntu/Debian: sudo apt-get install poppler-utils") 
        print("- Windows: Download and install poppler")
        return
    
    # Process all folders
    try:
        converter.process_all_folders(
            skip_existing=True,  # Set to False to overwrite existing files
            dpi=300  # High quality conversion
        )
    except KeyboardInterrupt:
        print("\nConversion interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()