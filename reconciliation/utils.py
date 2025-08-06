from django.shortcuts import get_object_or_404
from core.models import Company, AuditLog


def get_user_company(user):
    """Get the company associated with the user."""
    if hasattr(user, 'profile'):
        return user.profile.company
    raise ValueError("User does not have an associated company profile")


def create_audit_log(user, company, action, model_name, object_id, changes, ip_address=None, user_agent=None):
    """Create an audit log entry."""
    return AuditLog.objects.create(
        user=user,
        company=company,
        action=action,
        model_name=model_name,
        object_id=object_id,
        changes=changes,
        ip_address=ip_address,
        user_agent=user_agent
    )


def validate_csv_headers(headers, required_headers):
    """Validate that CSV has required headers."""
    missing_headers = set(required_headers) - set(headers)
    if missing_headers:
        return False, f"Missing required headers: {', '.join(missing_headers)}"
    return True, None


def clean_amount_string(amount_str):
    """Clean and convert amount string to decimal."""
    import re
    from decimal import Decimal, InvalidOperation
    
    if not amount_str or amount_str.strip() == '':
        return Decimal('0')
    
    # Remove currency symbols, commas, and spaces
    cleaned = re.sub(r'[£$€,\s]', '', str(amount_str))
    
    # Handle negative amounts in parentheses
    if cleaned.startswith('(') and cleaned.endswith(')'):
        cleaned = '-' + cleaned[1:-1]
    
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        raise ValueError(f"Invalid amount format: {amount_str}")


def extract_reference_numbers(description):
    """Extract potential reference numbers from transaction description."""
    import re
    
    # Common patterns for reference numbers
    patterns = [
        r'\b[A-Z]{2,4}\d{4,}\b',  # Letters followed by numbers
        r'\b\d{4,10}\b',  # Pure numeric references
        r'\b[A-Z0-9]{6,}\b',  # Mixed alphanumeric
        r'INV[-_]?\d+',  # Invoice patterns
        r'REF[-_]?\d+',  # Reference patterns
    ]
    
    references = []
    for pattern in patterns:
        matches = re.findall(pattern, description.upper())
        references.extend(matches)
    
    return list(set(references))  # Remove duplicates


def calculate_similarity_score(text1, text2):
    """Calculate text similarity score using different methods."""
    from difflib import SequenceMatcher
    import re
    
    # Clean texts
    clean_text1 = re.sub(r'[^\w\s]', '', text1.lower())
    clean_text2 = re.sub(r'[^\w\s]', '', text2.lower())
    
    # Calculate sequence similarity
    seq_similarity = SequenceMatcher(None, clean_text1, clean_text2).ratio()
    
    # Calculate word overlap
    words1 = set(clean_text1.split())
    words2 = set(clean_text2.split())
    word_overlap = len(words1.intersection(words2)) / max(len(words1.union(words2)), 1)
    
    # Weighted average
    return (seq_similarity * 0.6) + (word_overlap * 0.4)


def format_currency(amount, currency_code='USD'):
    """Format amount as currency string."""
    currency_symbols = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥',
    }
    
    symbol = currency_symbols.get(currency_code, currency_code)
    return f"{symbol}{amount:,.2f}"


def get_date_range_filter(request, field_name='created_at'):
    """Get date range filter from request parameters."""
    from django.utils.dateparse import parse_date
    
    start_date = request.query_params.get('start_date')
    end_date = request.query_params.get('end_date')
    
    filters = {}
    if start_date:
        parsed_start = parse_date(start_date)
        if parsed_start:
            filters[f'{field_name}__gte'] = parsed_start
    
    if end_date:
        parsed_end = parse_date(end_date)
        if parsed_end:
            filters[f'{field_name}__lte'] = parsed_end
    
    return filters
