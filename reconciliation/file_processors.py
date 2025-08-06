import csv
from datetime import datetime
from decimal import Decimal
from .utils import clean_amount_string, validate_csv_headers
import logging

logger = logging.getLogger(__name__)

# Try to import pandas, use CSV fallback if not available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("pandas not available, using CSV-only processing")


class BaseFileProcessor:
    """Base class for file processors."""
    
    REQUIRED_HEADERS = ['date', 'description', 'amount']
    OPTIONAL_HEADERS = ['reference', 'bank_reference', 'type', 'balance']
    
    def __init__(self):
        self.processed_data = []
        self.errors = []
    
    def process_file(self, file_path):
        """Process file and return list of transaction dictionaries."""
        raise NotImplementedError("Subclasses must implement process_file method")
    
    def validate_and_clean_row(self, row):
        """Validate and clean a single row of data."""
        try:
            # Clean and validate date
            date_str = str(row.get('date', '')).strip()
            if not date_str:
                raise ValueError("Date is required")
            
            # Try different date formats
            date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y']
            transaction_date = None
            
            for fmt in date_formats:
                try:
                    transaction_date = datetime.strptime(date_str, fmt).date()
                    break
                except ValueError:
                    continue
            
            if not transaction_date:
                raise ValueError(f"Invalid date format: {date_str}")
            
            # Clean and validate description
            description = str(row.get('description', '')).strip()
            if not description:
                raise ValueError("Description is required")
            
            # Clean and validate amount
            amount_str = str(row.get('amount', '')).strip()
            if not amount_str:
                raise ValueError("Amount is required")
            
            amount = clean_amount_string(amount_str)
            
            # Optional fields
            reference = str(row.get('reference', '')).strip()
            bank_reference = str(row.get('bank_reference', '')).strip()
            transaction_type = str(row.get('type', '')).strip()
            
            balance = None
            if row.get('balance'):
                balance = clean_amount_string(str(row.get('balance')))
            
            return {
                'date': transaction_date,
                'description': description,
                'amount': amount,
                'reference': reference,
                'bank_reference': bank_reference,
                'type': transaction_type,
                'balance': balance,
                'raw_row': dict(row)
            }
            
        except Exception as e:
            raise ValueError(f"Row validation failed: {e}")


class CSVProcessor(BaseFileProcessor):
    """Processor for CSV bank statement files."""
    
    def process_file(self, file_path):
        """Process CSV file."""
        processed_data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                # Try to detect delimiter
                sample = file.read(1024)
                file.seek(0)
                
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                reader = csv.DictReader(file, delimiter=delimiter)
                headers = [h.lower().strip() for h in reader.fieldnames]
                
                # Validate headers
                is_valid, error = validate_csv_headers(headers, self.REQUIRED_HEADERS)
                if not is_valid:
                    raise ValueError(error)
                
                row_number = 1
                for row in reader:
                    row_number += 1
                    try:
                        # Convert keys to lowercase
                        clean_row = {k.lower().strip(): v for k, v in row.items()}
                        validated_row = self.validate_and_clean_row(clean_row)
                        processed_data.append(validated_row)
                        
                    except Exception as e:
                        logger.warning(f"Row {row_number} skipped: {e}")
                        self.errors.append({
                            'row_number': row_number,
                            'error': str(e),
                            'raw_data': dict(row)
                        })
                
        except Exception as e:
            logger.error(f"CSV processing failed: {e}")
            raise
        
        logger.info(f"CSV processing completed. Processed: {len(processed_data)}, Errors: {len(self.errors)}")
        return processed_data


class ExcelProcessor(BaseFileProcessor):
    """Processor for Excel files (.xlsx, .xls)."""
    
    def process_file(self, file_path):
        """Process Excel file using pandas or openpyxl."""
        logger.info(f"Processing Excel file: {file_path}")
        
        if not HAS_PANDAS:
            # Use openpyxl for basic Excel processing
            return self._process_excel_with_openpyxl(file_path)
        
        try:
            # Read Excel file
            df = pd.read_excel(file_path)
            
            if df.empty:
                raise ValueError("Excel file is empty")
            
            # Clean column names
            df.columns = [col.lower().strip() for col in df.columns]
            
            # Validate headers
            is_valid, error = validate_csv_headers(df.columns.tolist(), self.REQUIRED_HEADERS)
            if not is_valid:
                raise ValueError(error)
            
            # Process each row
            processed_data = []
            for index, row in df.iterrows():
                try:
                    # Convert pandas row to dictionary
                    row_dict = row.to_dict()
                    
                    # Handle NaN values
                    for key, value in row_dict.items():
                        if pd.isna(value):
                            row_dict[key] = ''
                    
                    validated_row = self.validate_and_clean_row(row_dict)
                    processed_data.append(validated_row)
                    
                except Exception as e:
                    logger.warning(f"Row {index + 2} skipped: {e}")
                    self.errors.append({
                        'row_number': index + 2,
                        'error': str(e),
                        'raw_data': row_dict
                    })
        
        except Exception as e:
            logger.error(f"Excel processing failed: {e}")
            raise
        
        logger.info(f"Excel processing completed. Processed: {len(processed_data)}, Errors: {len(self.errors)}")
        return processed_data
    
    def _process_excel_with_openpyxl(self, file_path):
        """Process Excel file using openpyxl as fallback."""
        import openpyxl
        
        logger.info("Using openpyxl for Excel processing (pandas not available)")
        
        try:
            workbook = openpyxl.load_workbook(file_path, read_only=True)
            worksheet = workbook.active
            
            # Get headers from first row
            headers = []
            for cell in worksheet[1]:
                if cell.value:
                    headers.append(str(cell.value).lower().strip())
            
            # Validate headers
            is_valid, error = validate_csv_headers(headers, self.REQUIRED_HEADERS)
            if not is_valid:
                raise ValueError(error)
            
            processed_data = []
            # Process data rows
            for row_num, row in enumerate(worksheet.iter_rows(min_row=2, values_only=True), start=2):
                try:
                    # Create dictionary from row data
                    row_dict = {}
                    for i, value in enumerate(row):
                        if i < len(headers):
                            row_dict[headers[i]] = str(value) if value is not None else ''
                    
                    validated_row = self.validate_and_clean_row(row_dict)
                    processed_data.append(validated_row)
                    
                except Exception as e:
                    logger.warning(f"Row {row_num} skipped: {e}")
                    self.errors.append({
                        'row_number': row_num,
                        'error': str(e),
                        'raw_data': row_dict
                    })
            
            workbook.close()
            
        except Exception as e:
            logger.error(f"Excel processing with openpyxl failed: {e}")
            raise
        
        logger.info(f"Excel processing completed. Processed: {len(processed_data)}, Errors: {len(self.errors)}")
        return processed_data


class BankStatementValidator:
    """Validator for bank statement data quality."""
    
    def __init__(self):
        self.warnings = []
        self.errors = []
    
    def validate_transactions(self, transactions):
        """Validate processed transactions for data quality."""
        if not transactions:
            self.errors.append("No transactions found in file")
            return False
        
        # Check for duplicate transactions
        seen_transactions = set()
        duplicates = []
        
        for i, transaction in enumerate(transactions):
            transaction_key = (
                transaction['date'],
                transaction['description'],
                transaction['amount']
            )
            
            if transaction_key in seen_transactions:
                duplicates.append(i)
            else:
                seen_transactions.add(transaction_key)
        
        if duplicates:
            self.warnings.append(f"Found {len(duplicates)} potential duplicate transactions")
        
        # Check for missing reference numbers
        missing_refs = sum(1 for t in transactions if not t.get('reference'))
        if missing_refs > len(transactions) * 0.5:
            self.warnings.append(f"{missing_refs} transactions missing reference numbers")
        
        # Check date range consistency
        dates = [t['date'] for t in transactions]
        if dates:
            date_range = max(dates) - min(dates)
            if date_range.days > 365:
                self.warnings.append(f"Transaction date range spans {date_range.days} days")
        
        # Check for unusually large amounts
        amounts = [abs(t['amount']) for t in transactions]
        if amounts:
            avg_amount = sum(amounts) / len(amounts)
            large_amounts = [a for a in amounts if a > avg_amount * 10]
            
            if large_amounts:
                self.warnings.append(f"Found {len(large_amounts)} transactions with unusually large amounts")
        
        return len(self.errors) == 0
    
    def get_validation_report(self):
        """Get validation report."""
        return {
            'errors': self.errors,
            'warnings': self.warnings,
            'is_valid': len(self.errors) == 0
        }
