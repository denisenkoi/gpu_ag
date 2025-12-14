"""
Utility functions for PAPI data processing
"""

def extract_val(data):
    """
    Extract value from PAPI format.
    
    PAPI returns data in format {'val': 123.45} or {'undefined': true}
    This function extracts the actual value for use in JSON output.
    
    Args:
        data: PAPI data field (can be dict with 'val' key or direct value)
        
    Returns:
        Extracted numeric/string value or None if undefined
    """
    if data is None:
        return None
    if isinstance(data, dict):
        if data.get('undefined', False):
            return None
        if 'val' in data:
            return data['val']
    return data