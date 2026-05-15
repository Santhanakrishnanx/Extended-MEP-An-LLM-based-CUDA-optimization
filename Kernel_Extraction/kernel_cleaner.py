import re

def clean_kernels(kernels):
    valid = []
    for k in kernels:
        if k.count("{") == k.count("}") and k.strip().endswith("}"):
            if "threadIdx" not in k:
                continue
            
            if "__global__" not in k:
                continue
            
            if "blockIdx" not in k:
                continue
            valid.append(k)
        else:
            print("Dropped invalid kernel (incomplete)")
    return valid

def clean_polybench_kernel(kernel_code):
    replacements = {
        "DATA_TYPE": "float",
        "_PB_N": "n",
        "_PB_M": "m",
        "_PB_K": "k",
        "N": "n",
        "M": "m",
        "K": "k",
        "_PB_NI": "ni",
        "_PB_NJ": "nj",
        "_PB_NK": "nk",
        "_PB_NL": "nl",
        "_PB_NM": "nm",
        "NI": "ni",
        "NJ": "nj",
        "NK": "nk",
        "NL": "nl",
        "NM": "nm"
    }
    kernel_code = kernel_code.replace("[i][j]", "[i * stride + j]")
    for k, v in replacements.items():
        kernel_code = kernel_code.replace(k, v)

    return kernel_code

def remove_linux_headers(code):
    lines = code.split("\n")

    filtered = []
    for line in lines:
        if "unistd.h" in line:
            continue
        if "sys/time.h" in line:
            continue
        filtered.append(line)

    return "\n".join(filtered)