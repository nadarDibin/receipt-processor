#!/usr/bin/env python3
"""
Receipt Processing System
Extracts amounts and categorizes receipts from digital receipts/invoices.
Supports currency conversion from USD to INR.
"""

import argparse
import csv
import os
import re
import traceback
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union

# Core dependencies
try:
    import pytesseract
    from PIL import Image
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    print("Warning: pytesseract not installed. Install with: pip install pytesseract Pillow")

# Optional dependencies
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("Warning: PyMuPDF not installed. Install with: pip install PyMuPDF (for PDF support)")

try:
    from google.cloud import vision
    HAS_VISION_API = True
except ImportError:
    HAS_VISION_API = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class ReceiptProcessor:
    """Receipt processor that extracts amounts and categorizes expenses."""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf'}
    DEFAULT_USD_TO_INR = 83.0
    
    AMOUNT_PATTERNS = [
        r'₹\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
        r'Rs\.?\s*(\d+(?:,\d{3})*\.\d{1,2})', 
        r'INR\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
        r'\$\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
        r'(?<![a-zA-Z₹$])(\d+(?:,\d{3})*\.\d{1,2})(?![a-zA-Z])',
        r'(?<![a-zA-Z₹$])(\d+(?:,\d{3})+)(?![.\d])'
    ]
    
    SKIP_KEYWORDS = ['credit card', 'card ending', 'order number', 'invoice number', 'phone', 'mobile']
    TOTAL_KEYWORDS = ['grand total', 'total:', 'net total', 'amount due', 'subtotal', 'total', 'amount', 'grand']
    
    def __init__(self, use_vision_api: bool = False, debug: bool = False):
        """Initialize the receipt processor."""
        self.use_vision_api = use_vision_api
        self.debug = debug
        self.categories = self._load_categories()
        self.usd_to_inr_rate = self._get_exchange_rate()
        self.vision_client = self._init_vision_client()

    def _init_vision_client(self):
        """Initialize Google Vision API client if available."""
        if self.use_vision_api and HAS_VISION_API:
            return vision.ImageAnnotatorClient()
        elif self.use_vision_api:
            print("Warning: Vision API requested but not available")
        return None

    def _load_categories(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined expense categories."""
        return {
            "Parking Reimbursement": {
                "description": "Reimbursement for parking charges",
                "keywords": ["parking", "park", "wework", "office parking"]
            },
            "Repair & Maintenance": {
                "description": "Laptop repair or urgent travel purchases",
                "keywords": ["repair", "maintenance", "laptop", "fix", "service", "urgent"]
            },
            "Printing & Courier": {
                "description": "Stationary and printing expenses",
                "keywords": ["printing", "print", "courier", "stationary", "stationery", "meeting", "paper", "ink"]
            },
            "Workcation": {
                "description": "Workcation claims",
                "keywords": ["workcation", "workation", "remote work", "co-working"]
            },
            "Business Travel": {
                "description": "Travel - Domestic & Overseas",
                "keywords": ["travel", "flight", "hotel", "cab", "taxi", "uber", "ola", "train", "bus", 
                           "makemytrip", "booking", "airbnb", "goibibo", "yatra"]
            },
            "Learning & Development": {
                "description": "Books and courses (₹25,000 limit)",
                "keywords": ["book", "course", "learning", "training", "development", "education", 
                           "skill", "certification", "amazon", "kindle", "udemy", "coursera"]
            },
            "Home Workstation": {
                "description": "Home office setup (₹25,000 one-time)",
                "keywords": ["furniture", "desk", "chair", "monitor", "workstation", "home office", 
                           "setup", "table", "ergonomic"]
            },
            "Mobile Handset": {
                "description": "Mobile purchase (₹42,000 per 24 months)",
                "keywords": ["mobile", "phone", "smartphone", "handset", "iphone", "samsung", 
                           "oneplus", "xiaomi", "realme"]
            },
            "Software & Subscriptions": {
                "description": "Software licenses and subscriptions",
                "keywords": ["software", "license", "subscription", "claude", "chatgpt", "adobe", 
                           "microsoft", "google", "aws", "cloud"]
            }
        }

    def _get_exchange_rate(self) -> float:
        """Get USD to INR exchange rate with API fallback."""
        if not HAS_REQUESTS:
            self._debug_print(f"Using default exchange rate: {self.DEFAULT_USD_TO_INR}")
            return self.DEFAULT_USD_TO_INR
        
        try:
            response = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=5)
            if response.status_code == 200:
                rate = response.json()['rates'].get('INR', self.DEFAULT_USD_TO_INR)
                self._debug_print(f"Got live exchange rate: 1 USD = {rate} INR")
                return rate
        except Exception as e:
            self._debug_print(f"Exchange rate API failed: {e}")
        
        self._debug_print(f"Using default exchange rate: {self.DEFAULT_USD_TO_INR}")
        return self.DEFAULT_USD_TO_INR

    def _convert_to_inr(self, amount: float, currency: str) -> float:
        """Convert amount to INR if needed."""
        if currency.upper() == 'USD':
            converted = amount * self.usd_to_inr_rate
            self._debug_print(f"Converted ${amount} to ₹{converted:.2f} using rate {self.usd_to_inr_rate}")
            return converted
        return amount

    def _debug_print(self, message: str):
        """Print debug messages if enabled."""
        if self.debug:
            print(f"DEBUG: {message}")

    def extract_text(self, image_path: str) -> str:
        """Extract text from receipt using available OCR method."""
        # Try direct PDF text extraction first
        if str(image_path).lower().endswith('.pdf') and HAS_PYMUPDF:
            text = self._extract_pdf_text_direct(image_path)
            if text:
                return text

        # Try Vision API if available
        if self.vision_client:
            text = self._extract_text_vision_api(image_path)
            if text:
                return text

        # Fallback to Tesseract
        if HAS_TESSERACT:
            return self._extract_text_tesseract(image_path)
        
        raise ImportError("No OCR method available")

    def _extract_pdf_text_direct(self, pdf_path: str) -> str:
        """Extract text directly from PDF."""
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_parts.append(page.get_text())
            doc.close()
            
            full_text = '\n'.join(text_parts)
            if len(full_text.strip()) > 100:
                self._debug_print(f"Using direct PDF text extraction ({len(full_text)} chars)")
                return full_text
        except Exception as e:
            self._debug_print(f"Direct PDF extraction failed: {e}")
        return ""

    def _extract_text_vision_api(self, image_path: str) -> str:
        """Extract text using Google Vision API."""
        try:
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            response = self.vision_client.text_detection(image=image)
            texts = response.text_annotations
            
            return texts[0].description.strip() if texts else ""
        except Exception as e:
            self._debug_print(f"Vision API error: {e}")
            return ""

    def _extract_text_tesseract(self, image_path: str) -> str:
        """Extract text using Tesseract OCR."""
        try:
            if str(image_path).lower().endswith('.pdf'):
                return self._extract_pdf_with_ocr(image_path)
            
            # Process regular images
            image = Image.open(image_path)
            image = self._preprocess_image(image)
            
            # Try multiple PSM configurations
            configs = ['--psm 6', '--psm 3', '--psm 11']
            texts = []
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(image, lang='eng', config=config)
                    if text.strip():
                        texts.append(text)
                        self._debug_print(f"Config '{config}': Found {len(text)} chars")
                except Exception:
                    continue
            
            return max(texts, key=len) if texts else ""
        except Exception as e:
            self._debug_print(f"Tesseract error: {e}")
            return ""

    def _extract_pdf_with_ocr(self, pdf_path: str) -> str:
        """Extract text from PDF using OCR on converted images."""
        if not HAS_PYMUPDF:
            return ""
        
        try:
            doc = fitz.open(pdf_path)
            all_text = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                
                from io import BytesIO
                img_data = BytesIO(pix.tobytes("png"))
                image = Image.open(img_data)
                image = self._preprocess_image(image)
                
                text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
                all_text.append(text.strip())
            
            doc.close()
            return '\n'.join(all_text)
        except Exception as e:
            self._debug_print(f"PDF OCR error: {e}")
            return ""

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too small
        width, height = image.size
        if width < 1000:
            scale_factor = 1500 / width
            new_size = (int(width * scale_factor), int(height * scale_factor))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Apply OpenCV preprocessing if available
        if HAS_CV2:
            return self._opencv_preprocess(image)
        
        return image

    def _opencv_preprocess(self, image: Image.Image) -> Image.Image:
        """Apply OpenCV preprocessing for better OCR."""
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        
        # Resize if needed
        height, width = gray.shape
        if width < 1000:
            scale_factor = 1500 / width
            new_width, new_height = int(width * scale_factor), int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply adaptive thresholding and denoising
        processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processed = cv2.medianBlur(processed, 1)
        
        return Image.fromarray(processed)

    def extract_amount_from_line(self, line: str) -> List[float]:
        """Extract amounts from a single line with currency conversion."""
        amounts = set()
        found_positions = set()
        
        for pattern in self.AMOUNT_PATTERNS:
            for match in re.finditer(pattern, line):
                start, end = match.span(1)
                
                # Skip overlapping matches
                if any(start < pos[1] and end > pos[0] for pos in found_positions):
                    continue
                
                try:
                    amount_str = match.group(1).replace(',', '')
                    amount = float(amount_str)
                    
                    # Detect and convert currency
                    if '$' in line or 'USD' in line.upper():
                        amount = self._convert_to_inr(amount, 'USD')
                    
                    if 1 <= amount <= 100000:  # Reasonable range in INR
                        amounts.add(amount)
                        found_positions.add((start, end))
                except (ValueError, AttributeError):
                    continue
        
        return sorted(list(amounts))

    def extract_amount(self, text: str) -> Optional[float]:
        """Extract the most likely total amount from receipt text."""
        if not text:
            return None
        
        self._debug_print("Extracting amounts...")
        amounts_found = []
        lines = [' '.join(line.split()) for line in text.split('\n')]
        
        # Process each line
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line or any(keyword in line.lower() for keyword in self.SKIP_KEYWORDS):
                continue
            
            line_lower = line.lower()
            has_currency = any(symbol in line for symbol in ['₹', 'Rs', 'INR', '$'])
            is_total_line = any(keyword in line_lower for keyword in self.TOTAL_KEYWORDS)
            
            line_amounts = self.extract_amount_from_line(line)
            
            for amount in line_amounts:
                priority = self._get_amount_priority(line_lower, has_currency, is_total_line)
                amounts_found.append((amount, priority, line_num, line))
                self._debug_print(f"Found: ₹{amount} (priority {priority}) in line {line_num}: {line[:50]}...")
        
        # Check for amounts near total keywords
        amounts_found.extend(self._find_amounts_near_totals(lines))
        
        if not amounts_found:
            self._debug_print("No amounts found")
            return None
        
        return self._select_best_amount(amounts_found)

    def _get_amount_priority(self, line_lower: str, has_currency: bool, is_total_line: bool) -> int:
        """Determine priority for an amount based on context."""
        if 'grand total' in line_lower:
            return 5
        elif is_total_line:
            return 4
        elif has_currency:
            return 3
        else:
            return 1

    def _find_amounts_near_totals(self, lines: List[str]) -> List[Tuple[float, int, int, str]]:
        """Find amounts near total keywords within 2 lines."""
        nearby_amounts = []
        
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ['grand total', 'total:', 'amount due']):
                for offset in [-2, -1, 1, 2]:
                    if 0 <= i + offset < len(lines):
                        nearby_line = lines[i + offset]
                        amounts = self.extract_amount_from_line(nearby_line)
                        for amount in amounts:
                            nearby_amounts.append((amount, 4, i + offset, nearby_line))
                            self._debug_print(f"Found near total: ₹{amount} in line {i + offset}")
        
        return nearby_amounts

    def _select_best_amount(self, amounts_found: List[Tuple[float, int, int, str]]) -> float:
        """Select the best amount from candidates."""
        if self.debug and len(amounts_found) > 1:
            print("\nDEBUG: All amounts found (sorted by priority):")
            sorted_amounts = sorted(amounts_found, key=lambda x: (x[1], x[0]), reverse=True)
            for amt, pri, line_num, line_text in sorted_amounts[:5]:
                print(f"  Priority {pri}: ₹{amt} at line {line_num}: {line_text[:60]}...")
        
        # Sort by priority then amount
        amounts_found.sort(key=lambda x: (x[1], x[0]), reverse=True)
        selected = amounts_found[0]
        
        self._debug_print(f"Selected: ₹{selected[0]} from line {selected[2]}: {selected[3][:60]}...")
        
        # Validation for large amounts
        if selected[0] > 1000:
            high_priority_amounts = [a for a in amounts_found if a[1] >= 4 and a[0] < 1000]
            if high_priority_amounts and high_priority_amounts[0][1] == 5:  # Grand Total
                selected = high_priority_amounts[0]
                self._debug_print(f"Switched to: ₹{selected[0]} from line {selected[2]}")
        
        return selected[0]

    def categorize_receipt(self, text: str, amount: Optional[float] = None) -> Tuple[str, float]:
        """Categorize receipt based on text content."""
        if not text:
            return "Uncategorized", 0.0
        
        text_lower = text.lower()
        scores = {}
        
        for category, info in self.categories.items():
            score = sum(1 for keyword in info['keywords'] if keyword.lower() in text_lower)
            scores[category] = score
        
        if scores and max(scores.values()) > 0:
            best_category = max(scores, key=scores.get)
            confidence = scores[best_category] / len(self.categories[best_category]['keywords'])
            return best_category, confidence
        
        return "Uncategorized", 0.0

    def process_receipt(self, image_path: str) -> Dict[str, Any]:
        """Process a single receipt and extract information."""
        filename = os.path.basename(image_path)
        
        if self.debug:
            print(f"\n{'='*60}")
            print(f"Processing: {filename}")
            print('='*60)
        
        try:
            text = self.extract_text(image_path)
            
            if self.debug and text:
                print(f"\nExtracted text preview:\n{'-'*40}")
                print(text[:500] + ("..." if len(text) > 500 else ""))
                print('-'*40)
            
            if not text:
                return self._create_error_result(filename, 'No text extracted')
            
            amount = self.extract_amount(text)
            category, confidence = self.categorize_receipt(text, amount)
            
            return {
                'file': filename,
                'amount': amount,
                'category': category,
                'confidence': confidence,
                'error': None,
                'extracted_text': text[:200] + ('...' if len(text) > 200 else text)
            }
            
        except Exception as e:
            if self.debug:
                print(f"\nError: {e}")
                traceback.print_exc()
            return self._create_error_result(filename, str(e))

    def _create_error_result(self, filename: str, error: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'file': filename,
            'amount': None,
            'category': 'Error',
            'confidence': 0.0,
            'error': error,
            'extracted_text': ''
        }

    def process_batch(self, input_folder: str, output_csv: str):
        """Process all receipt images in a folder."""
        input_path = Path(input_folder)
        
        # Find supported files
        image_files = []
        for ext in self.SUPPORTED_FORMATS:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No supported files found in {input_folder}")
            print(f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}")
            return
        
        print(f"Found {len(image_files)} receipt files")
        print("-" * 50)
        
        results = []
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {image_file.name}")
            
            result = self.process_receipt(str(image_file))
            results.append(result)
            
            # Print summary
            status = "✓" if result['error'] is None else "✗"
            amount_str = f"₹{result['amount']:,.2f}" if result['amount'] else "Not found"
            print(f"{status} Amount: {amount_str} | Category: {result['category']} ({result['confidence']:.0%})")
        
        self.save_to_csv(results, output_csv)
        self._print_summary(results, output_csv)

    def _print_summary(self, results: List[Dict], output_csv: str):
        """Print processing summary."""
        successful = sum(1 for r in results if r['error'] is None)
        print(f"\n{'='*50}")
        print("PROCESSING COMPLETE")
        print(f"{'='*50}")
        print(f"✓ Successful: {successful}/{len(results)}")
        print(f"✗ Failed: {len(results) - successful}/{len(results)}")
        print(f"📁 Results saved to: {output_csv}")

    def save_to_csv(self, results: List[Dict], output_file: str):
        """Save processing results to CSV file."""
        if not results:
            print("No results to save")
            return
        
        fieldnames = ['file', 'amount', 'category', 'confidence', 'error', 'extracted_text']
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = result.copy()
                if row['amount'] is not None:
                    row['amount'] = f"{row['amount']:.2f}"
                row['confidence'] = f"{row['confidence']:.1%}"
                writer.writerow(row)


def main():
    """Main entry point for the receipt processor."""
    parser = argparse.ArgumentParser(
        description='Process receipt images and extract expense data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s receipts/                    Process all receipts in folder
  %(prog)s receipts/ -o expenses.csv    Save to specific file
  %(prog)s receipts/ --debug            Show detailed debug output
  %(prog)s receipts/ --vision-api       Use Google Vision API
        """
    )
    
    parser.add_argument('input_folder', help='Folder containing receipt images')
    parser.add_argument('-o', '--output', default='receipts_processed.csv',
                       help='Output CSV file (default: receipts_processed.csv)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--vision-api', action='store_true', 
                       help='Use Google Vision API (requires setup)')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.isdir(args.input_folder):
        print(f"Error: '{args.input_folder}' is not a valid directory")
        return 1
    
    # Check dependencies
    if not HAS_TESSERACT and not args.vision_api:
        print("Error: No OCR method available. Please install pytesseract or use --vision-api")
        return 1
    
    # Initialize and run processor
    try:
        processor = ReceiptProcessor(use_vision_api=args.vision_api, debug=args.debug)
        processor.process_batch(args.input_folder, args.output)
        return 0
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        return 1
    except Exception as e:
        print(f"\nFatal error: {e}")
        if args.debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())