import re

def extract_kernel_features(kernel_code):
    features = {}

    features["num_loops"] = kernel_code.count("for")

    features["has_strided_access"] = (
        "k * nj + j" in kernel_code or
        "k * nl + j" in kernel_code
    )

    features["uses_shared_memory"] = "__shared__" in kernel_code
    features["is_2d"] = ("threadIdx.y" in kernel_code)
    features["compute_intensity"] = kernel_code.count("*") / max(1, kernel_code.count("for"))
    features["memory_ops"] = kernel_code.count("[")

    if features["num_loops"] >= 2 and features["compute_intensity"] > 5:
        features["kernel_type"] = "compute_bound"
    elif features["memory_ops"] > features["compute_intensity"]:
        features["kernel_type"] = "memory_bound"
    else:
        features["kernel_type"] = "balanced"

    return features

def is_memory_bound(kernel_code):
    return (
        "for" in kernel_code and
        ("* nj + j" in kernel_code or "* nl + j" in kernel_code)
    )

