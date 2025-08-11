# Receipt Processor Test Specification

## Overview

This document outlines the comprehensive testing strategy for the Receipt Processor system to ensure reliable extraction of amounts and categorization from Indian digital receipts/invoices.

## Test Coverage Areas

### 1. Amount Extraction (`TestAmountExtraction`)

#### 1.1 Currency Symbol Recognition
- **₹ (Rupee) Symbol**: `₹135.59`, `₹7202.0`, `₹7,202.00`
- **$ (Dollar) Symbol**: `$23.60`, `$20.00`
- **Rs Formats**: `Rs. 1500.00`, `Rs 500`
- **INR Format**: `INR 2000.50`

#### 1.2 Decimal Format Variations
- Single decimal: `7202.0`
- Double decimal: `7202.00`
- Comma separators: `1,234.56`
- Whole numbers: `50` (validated range)

#### 1.3 Amount Range Validation
- **Minimum**: ₹1.00 (valid)
- **Maximum**: ₹100,000.00 (valid)
- **Below minimum**: ₹0.50 (rejected)
- **Above maximum**: ₹150,000.00 (rejected)

#### 1.4 Priority System Testing
- **Priority 5**: Grand Total lines
- **Priority 4**: Total/Amount Due lines
- **Priority 3**: Currency symbol present
- **Priority 1**: Plain numbers

### 2. Receipt Categorization (`TestCategorization`)

#### 2.1 Business Travel
- **Keywords**: makemytrip, flight, hotel, taxi, booking
- **Expected confidence**: >40%
- **Sample text**: MakeMyTrip booking confirmations

#### 2.2 Software & Subscriptions
- **Keywords**: claude, software, subscription, license
- **Sample text**: Anthropic Claude Pro invoices

#### 2.3 Other Categories
- Parking Reimbursement
- Repair & Maintenance
- Learning & Development
- Home Workstation
- Mobile Handset

### 3. PDF Processing (`TestPDFProcessing`)

#### 3.1 Direct Text Extraction
- **Primary method**: PyMuPDF direct text extraction
- **Preserves**: Currency symbols (₹, $)
- **Performance**: Fast, accurate for digital PDFs

#### 3.2 OCR Fallback
- **Triggered when**: Direct extraction yields <100 characters
- **Method**: Tesseract OCR with image preprocessing
- **Fallback chain**: Vision API → Tesseract

### 4. Known Receipt Validation (`TestKnownReceiptValidation`)

#### 4.1 MakeMyTrip Receipt NF90196357902608.pdf
```
Expected Values:
- Amount: ₹7,202.00
- Category: Business Travel
- Confidence: >40%
- Key text: "MAKEMYTRIP", "Grand Total"
```

#### 4.2 MakeMyTrip Receipt NF90163357160940.pdf
```
Expected Values:
- Amount: ₹6,950.00
- Category: Business Travel
- Confidence: >40%
```

#### 4.3 Anthropic Invoice STD6FRKW-0001.pdf
```
Expected Values:
- Amount: $23.60
- Category: Software & Subscriptions
- Key text: "Anthropic", "Claude Pro"
```

### 5. Edge Cases (`TestEdgeCases`)

#### 5.1 Error Conditions
- Empty files
- Non-existent file paths
- Corrupted PDFs
- No text extraction possible

#### 5.2 Boundary Conditions
- No amounts found in text
- Multiple currencies in same receipt
- Malformed amount formats

### 6. Batch Processing (`TestBatchProcessing`)

#### 6.1 Directory Processing
- Process all supported formats in directory
- Generate CSV output with correct headers
- Handle mixed file types

#### 6.2 CSV Output Validation
- Correct column headers
- Proper amount formatting
- Error handling for failed receipts

## Test Execution

### Running Tests

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
python test_receipt_processor.py

# Or with pytest (if installed)
pytest test_receipt_processor.py -v

# Run specific test class
python -m unittest test_receipt_processor.TestKnownReceiptValidation -v

# Run with coverage
pytest --cov=complete_receipt_processor_v1 --cov-report=html
```

### Test Data Requirements

The tests expect the following files in the `receipts/` directory:
- `NF90196357902608.pdf` - MakeMyTrip invoice (₹7,202.00)
- `NF90163357160940.pdf` - MakeMyTrip invoice (₹6,950.00)
- `Invoice-STD6FRKW-0001.pdf` - Anthropic invoice ($23.60)

### Expected Test Results

All tests should pass with the current implementation:
- **Amount extraction**: 100% accurate for known formats
- **Currency preservation**: ₹ and $ symbols maintained
- **Categorization**: >40% confidence for business travel
- **Known receipts**: Exact amount matches

## Regression Testing

### Critical Test Cases

1. **Amount Extraction Regression**
   - Verify ₹7,202.00 extraction from NF90196357902608.pdf
   - Confirm currency symbol preservation
   - Validate Grand Total priority over Service Fees

2. **PDF Processing Regression**
   - Direct text extraction preferred over OCR
   - Fallback to OCR when direct extraction fails
   - Character count threshold (100 chars) working

3. **Categorization Stability**
   - Business travel detection for MakeMyTrip
   - Software subscription detection for Anthropic
   - Keyword matching accuracy

### Performance Benchmarks

- **Direct PDF extraction**: <1 second per receipt
- **OCR fallback**: <5 seconds per receipt
- **Batch processing**: <2 seconds per receipt average

## Continuous Integration

### Pre-commit Hooks
```bash
# Add to .git/hooks/pre-commit
python test_receipt_processor.py
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
```

### GitHub Actions (Recommended)
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    - name: Install dependencies
      run: |
        pip install -r requirements-dev.txt
    - name: Run tests
      run: python test_receipt_processor.py
```

## Maintenance

### Adding New Test Cases
1. Add test method to appropriate test class
2. Follow naming convention: `test_description_of_test`
3. Use `self.subTest()` for parameterized tests
4. Update this specification document

### Updating Expected Values
When receipt processing logic changes:
1. Update expected values in `TestKnownReceiptValidation`
2. Re-run tests to ensure they pass
3. Update test specification documentation
4. Commit changes together with code updates

## Quality Gates

Before releasing any changes to the receipt processor:
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Known receipt validation passes
- [ ] No regression in amount extraction accuracy
- [ ] Currency symbol preservation verified
- [ ] Performance benchmarks met