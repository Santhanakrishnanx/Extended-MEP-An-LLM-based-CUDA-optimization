import re
from kernel_extractor import extract_all_kernels
from AER.refinement import refine_with_llm
from Candidate_Generation import validate_kernel

def extract_program_context(cuda_code, kernels):
    context = {}

    if "for" in cuda_code and "*" in cuda_code:
        context["kernel_type"] = "matrix_multiplication"
    else:
        context["kernel_type"] = "unknown"

    if "k * nj + j" in cuda_code or "k * nl + j" in cuda_code:
        context["memory_pattern"] = "strided"
    else:
        context["memory_pattern"] = "coalesced"

    context["data_flow"] = "input → compute → output"
    context["dependency"] = "multi_kernel_pipeline" if len(kernels) > 1 else "standalone"

    return context

def select_hotspot_kernel(kernels):
    if not kernels:
        return None
    kernels_sorted = sorted(kernels, key=lambda k: len(k["code"]), reverse=True)

    return kernels_sorted[0]

def extract_hotspot_kernel(cuda_code, use_llm=False):
    kernels = extract_all_kernels(cuda_code)
    program_context = extract_program_context(cuda_code, kernels)

    if not kernels:
        raise ValueError("No CUDA kernels found")

    # Step 1: heuristic selection
    selected = select_hotspot_kernel(kernels)

    # Step 2: optional LLM refinement
    if use_llm and len(kernels) > 1:
        llm_selected = refine_with_llm(cuda_code, kernels)
        if llm_selected:
            selected = llm_selected

    # Step 3: validation
    if not validate_kernel(selected):
        raise ValueError("Extracted kernel is invalid")

    return {
        "kernel_name": selected["name"],
        "kernel_code": selected["code"],
        "total_kernels_found": len(kernels)
    }