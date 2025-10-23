"""Hallucination detection and validation for AI Vision OCR"""

import hashlib
from datetime import datetime
from typing import List, Tuple
from PIL import Image

class HallucinationDetector:
    """
    Detects and prevents hallucinations in OCR output by implementing
    multiple validation strategies and quality checks.
    """
    
    def __init__(self, 
                 max_length: int = 20000, 
                 min_confidence: float = 0.85,
                 min_dpi: int = 200):
        self.max_length = max_length
        self.min_confidence = min_confidence
        self.min_dpi = min_dpi
        self.required_elements = [
            "South Carolina",
            "State of South Carolina",
            "County",
            "sworn",
            "witness"
        ]

    def detect_repetitions(self, text: str, min_length: int = 100) -> List[str]:
        """
        Detect repeated blocks of text and return unique sections.
        
        Args:
            text (str): Text to analyze
            min_length (int): Minimum length of text block to consider
            
        Returns:
            List[str]: List of unique text blocks
        """
        paragraphs = text.split('\n\n')
        seen = set()
        unique_paragraphs = []
        
        for p in paragraphs:
            if len(p.strip()) > min_length:
                # Create hash of normalized text to detect similar paragraphs
                p_normalized = ' '.join(p.strip().lower().split())
                p_hash = hashlib.md5(p_normalized.encode()).hexdigest()
                if p_hash not in seen:
                    seen.add(p_hash)
                    unique_paragraphs.append(p)
                    
        return unique_paragraphs

    def validate_length(self, text: str) -> bool:
        """
        Check if transcribed text exceeds reasonable length.
        
        Args:
            text (str): Text to validate
            
        Returns:
            bool: True if length is within limits
        """
        return len(text) <= self.max_length

    def validate_structure(self, text: str) -> Tuple[bool, List[str]]:
        """
        Verify text has expected legal document structure.
        
        Args:
            text (str): Text to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, missing_elements)
        """
        text_lower = text.lower()
        missing = [el for el in self.required_elements 
                  if el.lower() not in text_lower]
        return len(missing) == 0, missing

    def check_image_quality(self, image_path: str) -> Tuple[bool, float]:
        """
        Verify image meets minimum quality requirements.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            Tuple[bool, float]: (meets_requirements, calculated_dpi)
        """
        try:
            img = Image.open(image_path)
            width, height = img.size
            
            # Calculate effective DPI (assuming standard page width of 8.5 inches)
            dpi = min(width, height) / 8.5
            return dpi >= self.min_dpi, dpi
            
        except Exception as e:
            print(f"Warning: Could not check image quality: {e}")
            return False, 0.0

    def analyze_text(self, text: str, image_path: str) -> dict:
        """
        Analyze text for potential hallucinations and quality issues.
        
        Args:
            text (str): Text to analyze
            image_path (str): Path to source image
            
        Returns:
            dict: Analysis results with flags and metrics
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "source_image": image_path,
            "text_length": len(text),
            "needs_review": False,
            "review_reasons": [],
            "quality_metrics": {}
        }
        
        # Check text length
        if not self.validate_length(text):
            results["review_reasons"].append(
                f"Text length ({len(text)}) exceeds maximum ({self.max_length})"
            )
            
        # Check document structure
        is_valid_structure, missing = self.validate_structure(text)
        if not is_valid_structure:
            results["review_reasons"].append(
                f"Missing expected elements: {', '.join(missing)}"
            )
            
        # Check for repetitions
        unique_blocks = self.detect_repetitions(text)
        if len(unique_blocks) < text.count('\n\n'):
            results["review_reasons"].append(
                f"Found {text.count('\n\n') - len(unique_blocks)} repeated text blocks"
            )
        
        # Check image quality
        meets_quality, dpi = self.check_image_quality(image_path)
        results["quality_metrics"]["dpi"] = dpi
        if not meets_quality:
            results["review_reasons"].append(
                f"Image DPI ({dpi:.1f}) below minimum ({self.min_dpi})"
            )
            
        results["needs_review"] = len(results["review_reasons"]) > 0
        return results

    def clean_hallucinations(self, text: str) -> str:
        """
        Remove detected hallucinations from text.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text with duplicates removed
        """
        # Get unique paragraphs
        unique_paragraphs = self.detect_repetitions(text)
        
        # If we found duplicates, use only unique content
        if len(unique_paragraphs) < text.count('\n\n'):
            return '\n\n'.join(unique_paragraphs)
            
        return text