#!/usr/bin/env python3
"""
Receipt Processing System
Extracts amount and categorizes receipts from Indian digital receipts/invoices
"""

import os
import csv
import json
import re
from datetime import datetime
from pathlib import Path
import argparse

# OCR Libraries
try:
    import pytesseract
    from PIL import Image
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    print("Warning: pytesseract not installed. Install with: pip install pytesseract Pillow")

# PDF processing
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("Warning: PyMuPDF not installed. Install with: pip install PyMuPDF (for PDF support)")

# Optional: Google Vision API for complex receipts
try:
    from google.cloud import vision
    HAS_VISION_API = True
except ImportError:
    HAS_VISION_API = False

# For AI categorization (local option)
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

class ReceiptProcessor:
    def __init__(self, use_vision_api=False):
        self.use_vision_api = use_vision_api
        self.categories = self._load_categories()
        
        # Initialize Vision API client if available
        if use_vision_api and HAS_VISION_API:
            self.vision_client = vision.ImageAnnotatorClient()
        else:
            self.vision_client = None
    
    def _load_categories(self):
        """Load predefined expense categories"""
        return {
            "Parking Reimbursement": {
                "description": "Reimbursement for parking charges alone - Exp - Wework office parking charges",
                "keywords": ["parking", "park", "wework", "office parking"]
            },
            "Repair & Maintenance": {
                "description": "Any laptop repair or charges purchased on urgent scenarios during travel",
                "keywords": ["repair", "maintenance", "laptop", "fix", "service", "urgent"]
            },
            "Printing & courier": {
                "description": "Expenses related to stationary items for meeting purposes",
                "keywords": ["printing", "print", "courier", "stationary", "stationery", "meeting", "paper", "ink"]
            },
            "Workcation": {
                "description": "All Workcation Claim",
                "keywords": ["workcation", "workation", "remote work", "co-working"]
            },
            "Business travel": {
                "description": "Travel - Both Domestic & Overseas",
                "keywords": ["travel", "flight", "hotel", "cab", "taxi", "uber", "ola", "train", "bus", "makemytrip", "booking", "airbnb"]
            },
            "Learning & development": {
                "description": "Books purchased, learning and development for employee (25000 INR)",
                "keywords": ["book", "course", "learning", "training", "development", "education", "skill", "certification", "amazon", "kindle"]
            },
            "Home workstation": {
                "description": "Furniture or office setup in home - one time for an employee (25000 INR)",
                "keywords": ["furniture", "desk", "chair", "monitor", "workstation", "home office", "setup"]
            },
            "Mobile Handset": {
                "description": "Mobile purchase reimbursement for 42000 INR - 24 months once",
                "keywords": ["mobile", "phone", "smartphone", "handset", "iphone", "samsung", "oneplus"]
            }
        }
    
    def pdf_to_images(self, pdf_path):
        """Convert PDF to images for OCR"""
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF not available for PDF processing")
        
        try:
            doc = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Convert to image with high DPI for better OCR
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                from io import BytesIO
                image = Image.open(BytesIO(img_data))
                images.append(image)
            
            doc.close()
            return images
        
        except Exception as e:
            print(f"Error converting PDF: {e}")
            return []

    def extract_text_tesseract(self, image_path):
        """Extract text using Tesseract OCR"""
        if not HAS_TESSERACT:
            raise ImportError("Tesseract not available")
        
        try:
            # Check if it's a PDF
            if str(image_path).lower().endswith('.pdf'):
                if not HAS_PYMUPDF:
                    return "PDF processing requires PyMuPDF: pip install PyMuPDF"
                
                # Convert PDF to images and extract text from all pages
                images = self.pdf_to_images(image_path)
                all_text = []
                
                for image in images:
                    # Convert to RGB if necessary
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Extract text with better config for receipts
                    text = pytesseract.image_to_string(image, lang='eng')
                    all_text.append(text.strip())
                
                return '\n'.join(all_text)
            
            else:
                # Regular image processing
                image = Image.open(image_path)
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Extract text
                text = pytesseract.image_to_string(image, lang='eng')
                return text.strip()
        
        except Exception as e:
            print(f"Error with Tesseract OCR: {e}")
            return ""
    
    def extract_text_vision_api(self, image_path):
        """Extract text using Google Vision API"""
        if not self.vision_client:
            raise ImportError("Vision API not available")
        
        try:
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            response = self.vision_client.text_detection(image=image)
            texts = response.text_annotations
            
            if texts:
                return texts[0].description.strip()
            return ""
        
        except Exception as e:
            print(f"Error with Vision API: {e}")
            return ""
    
    def extract_text(self, image_path):
        """Extract text from receipt image"""
        # Try Vision API first if available, fallback to Tesseract
        if self.use_vision_api and self.vision_client:
            text = self.extract_text_vision_api(image_path)
            if text:
                return text
        
        # Fallback to Tesseract
        if HAS_TESSERACT:
            return self.extract_text_tesseract(image_path)
        
        raise ImportError("No OCR method available. Install pytesseract or setup Google Vision API")
    
    def extract_amount(self, text):
        """Extract amount from receipt text"""
        print(f"DEBUG - Full text:\n{text}\n" + "="*50)
        
        amounts_found = []
        lines = text.split('\n')
        
        # Look for currency amounts in each line
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            print(f"DEBUG - Line {line_num}: {line}")
            
            # Skip lines that are clearly not about money
            line_lower = line.lower()
            skip_keywords = ['credit card', 'card ending', 'order number', 'invoice number', 'phone', 'mobile']
            if any(keyword in line_lower for keyword in skip_keywords):
                print(f"  -> Skipped (blacklist)")
                continue
            
            # Check for currency indicators
            has_rupee = '₹' in line
            has_rs = 'rs.' in line_lower or 'rs ' in line_lower
            has_inr = 'inr' in line_lower
            is_total_line = any(keyword in line_lower for keyword in ['grand total', 'total:', 'net total', 'amount due', 'subtotal'])
            
            print(f"  -> Rupee: {has_rupee}, Rs: {has_rs}, INR: {has_inr}, Total: {is_total_line}")
            
            # Only process lines with money context
            if not (has_rupee or has_rs or has_inr or is_total_line):
                print(f"  -> Skipped (no money context)")
                continue
            
            # Find all decimal numbers in the line
            # Look for patterns like: 236.28, 1,234.56, 50, 100.00
            import re
            number_patterns = [
                r'\d{1,6}(?:,\d{3})*\.\d{2}',  # 236.28, 1,234.56
                r'\d{1,6}(?:,\d{3})*\.\d{1}',  # 236.2
                r'\d{1,4}\.\d{2}',             # 50.00
                r'\d{1,6}(?:,\d{3})*'          # 236, 1,234 (whole numbers)
            ]
            
            found_numbers = []
            for pattern in number_patterns:
                matches = re.findall(pattern, line)
                found_numbers.extend(matches)
            
            print(f"  -> Numbers found: {found_numbers}")
            
            for num_str in found_numbers:
                try:
                    # Convert to float
                    clean_num = num_str.replace(',', '')
                    amount = float(clean_num)
                    
                    # Filter reasonable amounts
                    if amount < 1 or amount > 50000:
                        print(f"    -> {amount} out of range")
                        continue
                    
                    # Skip if it looks like a date, phone number, etc.
                    if len(clean_num) == 4 and amount > 1900 and amount < 2100:  # Year
                        print(f"    -> {amount} looks like year")
                        continue
                    if len(clean_num) > 8:  # Too long
                        print(f"    -> {amount} too long")
                        continue
                    
                    priority = 0
                    if is_total_line:
                        priority = 3
                    elif has_rupee:
                        priority = 2
                    elif has_rs or has_inr:
                        priority = 1
                    
                    amounts_found.append((amount, priority, line))
                    print(f"    -> VALID: ₹{amount} (priority: {priority})")
                
                except ValueError:
                    print(f"    -> Failed to convert: {num_str}")
                    continue
        
        print(f"DEBUG - All amounts found: {[(a[0], a[1]) for a in amounts_found]}")
        
        # Sort by priority, then amount
        amounts_found.sort(key=lambda x: (x[1], x[0]), reverse=True)
        
        if amounts_found:
            selected_amount = amounts_found[0][0]
            print(f"DEBUG - SELECTED: ₹{selected_amount}")
            return selected_amount
        
        print("DEBUG - NO AMOUNT FOUND")
        return None
    
    def categorize_receipt(self, text, amount=None):
        """Categorize receipt based on text content"""
        text_lower = text.lower()
        
        # Score each category based on keyword matches
        scores = {}
        for category, info in self.categories.items():
            score = 0
            for keyword in info['keywords']:
                if keyword.lower() in text_lower:
                    score += 1
            scores[category] = score
        
        # Get the best matching category
        if scores and max(scores.values()) > 0:
            best_category = max(scores.keys(), key=lambda k: scores[k])
            confidence = scores[best_category] / len(self.categories[best_category]['keywords'])
            return best_category, confidence
        
        return "Uncategorized", 0.0
    
    def process_receipt(self, image_path):
        """Process a single receipt"""
        print(f"Processing: {image_path}")
        
        try:
            # Extract text
            text = self.extract_text(image_path)
            if not text:
                return {
                    'file': os.path.basename(image_path),
                    'amount': None,
                    'category': 'Error',
                    'confidence': 0.0,
                    'error': 'No text extracted'
                }
            
            # Extract amount
            amount = self.extract_amount(text)
            
            # Categorize
            category, confidence = self.categorize_receipt(text, amount)
            
            result = {
                'file': os.path.basename(image_path),
                'amount': amount,
                'category': category,
                'confidence': confidence,
                'extracted_text': text[:200] + '...' if len(text) > 200 else text,
                'error': None
            }
            
            return result
            
        except Exception as e:
            return {
                'file': os.path.basename(image_path),
                'amount': None,
                'category': 'Error',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def process_batch(self, input_folder, output_csv):
        """Process all receipt images in a folder"""
        input_path = Path(input_folder)
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf'}
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No image files found in {input_folder}")
            return
        
        print(f"Found {len(image_files)} receipt files")
        
        results = []
        for image_file in image_files:
            result = self.process_receipt(str(image_file))
            results.append(result)
            
            # Print progress
            status = "✓" if result['error'] is None else "✗"
            print(f"{status} {result['file']}: ₹{result['amount']} - {result['category']} ({result['confidence']:.1%})")
        
        # Save to CSV
        self.save_to_csv(results, output_csv)
        print(f"\nResults saved to: {output_csv}")
        
        # Print summary
        successful = len([r for r in results if r['error'] is None])
        print(f"\nSummary: {successful}/{len(results)} receipts processed successfully")
    
    def save_to_csv(self, results, output_file):
        """Save results to CSV file"""
        fieldnames = ['file', 'amount', 'category', 'confidence', 'error', 'extracted_text']
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                # Format amount
                if result['amount']:
                    result['amount'] = f"₹{result['amount']:,.2f}"
                
                # Format confidence as percentage
                result['confidence'] = f"{result['confidence']:.1%}"
                
                writer.writerow(result)

def main():
    parser = argparse.ArgumentParser(description='Process receipt images and extract expense data')
    parser.add_argument('input_folder', help='Folder containing receipt images')
    parser.add_argument('-o', '--output', default='receipts_processed.csv', 
                       help='Output CSV file (default: receipts_processed.csv)')
    parser.add_argument('--vision-api', action='store_true', 
                       help='Use Google Vision API (requires setup)')
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.isdir(args.input_folder):
        print(f"Error: {args.input_folder} is not a valid directory")
        return
    
    # Initialize processor
    processor = ReceiptProcessor(use_vision_api=args.vision_api)
    
    # Process receipts
    processor.process_batch(args.input_folder, args.output)

if __name__ == "__main__":
    main()
