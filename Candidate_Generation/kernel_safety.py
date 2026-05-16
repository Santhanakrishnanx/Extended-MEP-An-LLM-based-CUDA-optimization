import re

def is_safe_kernel(code):
     return (
        "if" in code and
        ("row < n" in code or "col < n" in code)
    )
    
def reject_bad_kernel(code):
    unsafe_patterns = [
        "float2*",
        "float4*",
        "reinterpret_cast<float2",
        "reinterpret_cast<float4"
    ]
    dangerous_patterns = [
        "(c11 + c21 + c31)",
        "(c13 + c23 + c33)",
        "coeff_sum",
        "left_sum",
        "right_sum"
    ]
    
    if any(p in code for p in dangerous_patterns):
        print("Rejected (algebraic rewrite)")
        return True
    
    if any(p in code for p in unsafe_patterns):
        return True

    return False
    
def has_safe_tiling(kernel_code):
    # Reject kernels with fixed TILE loop
    if "for (int k = 0; k < TILE" in kernel_code:
        if "tileLimit" not in kernel_code and "limit" not in kernel_code:
            return False
    return True

def validate_indexing(kernel_code):
    rules = [
        ("A", "* nk +"),
        ("B", "* nj +"),
        ("C", "* nm +"),
        ("D", "* nl +"),
        ("E", "* nj +"),
        ("F", "* nl +"),
        ("G", "* nl +"),
    ]

    for var, pattern in rules:
        if var in kernel_code and pattern not in kernel_code:
            return False

    return True