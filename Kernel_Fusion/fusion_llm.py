from LLM_API.openai_client import client
from Kernel_Extraction.kernel_extractor import extract_complete_kernels

def fuse_kernels_llm(code1, code2, k1, k2):
    prompt = f"""
    You are given two CUDA kernels with dependency.
    
    KERNEL 1:
    {code1}
    
    KERNEL 2:
    {code2}
    
    DEPENDENCY:
    - Output of Kernel1 is used in Kernel2
    
    TASK:
    Generate a HIGH-PERFORMANCE fused kernel.
    
    CRITICAL REQUIREMENTS:
    
    1. DO NOT recompute intermediate values repeatedly
       - Intermediate values MUST be computed ONCE and reused
    
    2. Maintain computational complexity O(n^3)
       - NEVER introduce nested recomputation (O(n^4))
    
    3. USE SHARED MEMORY TILING
       - Tile A, B, and intermediate values
       - Reuse data across threads
    
    4. FUSION STRATEGY:
       - Compute tiles of E = A × B
       - Keep E in shared memory or registers
       - Immediately use it to compute final output
    
    5. MEMORY OPTIMIZATION:
       - Eliminate global memory writes for intermediate arrays
       - Use coalesced access
       - Use float4 if possible
    
    6. LOOP STRUCTURE:
       - DO NOT nest full matrix multiplication inside another
       - Use tiled multiplication pattern
    
    7. CORRECTNESS:
       - Must match original computation exactly
       - Do NOT replace arrays with others
       - Do NOT reinterpret inputs
    
    OUTPUT:
    Return ONLY the fused CUDA kernel
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2000
    )

    text = response.choices[0].message.content

    kernels = extract_complete_kernels(text)

    if not kernels:
        return None

    fused = kernels[0]

    if "__global__" not in fused:
        return None

    return fused

def reject_bad_fusion(kernel_code):
    if kernel_code is None:
        return True

    # detect O(n^4) pattern
    if kernel_code.count("for") >= 3:
        print("Rejected fusion (likely recomputation O(n^4))")
        return True

    # detect recomputation pattern
    if "E_val" in kernel_code and kernel_code.count("E_val") > 2:
        print("Rejected fusion (recomputing intermediate)")
        return True

    return False