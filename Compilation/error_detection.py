import re

def detect_cuda_error(output):
    return "CUDA_ERROR" in output