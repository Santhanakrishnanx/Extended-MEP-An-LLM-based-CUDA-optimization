import re

def validate_kernel(kernel):
    if not kernel or not kernel.get("code"):
        return False

    code = kernel["code"]

    # Check braces balance
    if code.count("{") != code.count("}"):
        return False

    # Check __global__ presence
    if "__global__" not in code:
        return False

    return True

def is_valid_cuda_kernel(code):
    if "__global__" not in code:
        return False

    if code.count("{") != code.count("}"):
        return False

    return True