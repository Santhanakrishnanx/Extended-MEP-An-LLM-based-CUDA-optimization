import re

def extract_checksum(output):
    import re
    match = re.search(r"CHECKSUM:\s*([0-9.eE+-]+)", output)
    return float(match.group(1)) if match else None

def extract_ref_checksum(output):
    match = re.search(r"REF_CHECKSUM:\s*([0-9.eE+-]+)", output)
    return float(match.group(1)) if match else None
    
def check_functional_equivalence(base_output, new_output):
    base_val = extract_checksum(base_output)
    new_val = extract_checksum(new_output)

    if base_val is None or new_val is None:
        return False

    diff = abs(base_val - new_val) / max(abs(base_val), 1e-8)

    return diff < 5e-2