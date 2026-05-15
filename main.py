from LLM_API.openai_client import client
from Utils.file_utils import read_cuda_file
from Kernel_Extraction.kernel_extractor import extract_all_kernels
from Candidate_Generation.candidate_generator import optimization_loop
from Validation.save_base import save_optimized_file
from Kernel_Extraction.kernel_cleaner import remove_linux_headers
from Kernel_Fusion.fusion_preprocess import fusion_preprocess
from Kernel_Extraction.hotspot_detector import extract_program_context
from Kernel_Extraction.kernel_cleaner import clean_polybench_kernel
from Candidate_Generation.candidate_generator import generate_mep
from Compilation.compiler import compile_run_with_repair
from Compilation.execution import evaluate_performance
from Validation.equivalence import extract_checksum
from Kernel_Fusion.fusion_preprocess import replace_kernel_in_code
from Candidate_Generation.kernel_validation import is_valid_cuda_kernel

def main():
    file_path = r"E:\GPU Kernel Optimization\polybenchGpu-master\polybenchGpu-master\CUDA\JACOBI2D\jacobi2D.cu"

    print("\n[STEP 1] Reading CUDA file...\n")
    cuda_code = read_cuda_file(file_path)
    cuda_code = remove_linux_headers(cuda_code)

    print("[STEP 2] Extracting hotspot kernel...\n")
    kernels = extract_all_kernels(cuda_code)
    program_context = extract_program_context(cuda_code, kernels)
    fused_kernels, fused_code = fusion_preprocess(cuda_code, kernels)
    
    kernels = fused_kernels
    cuda_code = fused_code
    program_context["original_code"] = cuda_code
    working_code = cuda_code
    optimized_kernels = {}
    for k in kernels:
        kernel_name = k["name"]
        kernel_code = clean_polybench_kernel(k["code"])
        program_context["kernel_name"] = kernel_name
    
        print(f"\n=== Optimizing {kernel_name} ===")
    
        if not kernel_code:
            print("Kernel extraction failed")
            return
    
        print("[KERNEL NAME]:", kernel_name, "\n")
        print("[STEP 3] Generating baseline MEP...\n")
        mep_code = generate_mep(kernel_code)
        print("\n========== GENERATED MEP ==========\n")
        print(mep_code)
        print("\n==================================\n")
        print("[STEP 4] Running baseline with AER...\n")
        baseline_result, fixed_mep = compile_run_with_repair(mep_code)
    
        if not baseline_result["run_success"]:
            print("Baseline failed even after repair")
            return
    
        baseline_output = baseline_result["runtime_output"]
        baseline_time = evaluate_performance(fixed_mep)
        if baseline_time is None:
            print("Baseline performance evaluation failed")
            continue
        baseline_checksum = extract_checksum(baseline_output)
    
        print(f"\n[BASELINE TIME]: {baseline_time} ms")
        print("\n[STEP 5] Optimization Loop...\n")
    
        best_kernel, best_time = optimization_loop(
            kernel_code,
            baseline_time,
            baseline_output,
            baseline_checksum,
            program_context,
            D=4,
            N=5
        )
    
        print(f"\n[FINAL BEST TIME]: {best_time} ms")
    
        if best_time:
            speedup = baseline_time / best_time
            print(f"[SPEEDUP]: {speedup:.2f}x")
        print("\n[STEP 6] Replacing kernel in original code...\n")
        print("\n[FINAL SELECTED KERNEL]\n")
        print(best_kernel)
        print("\n----------------------------------\n")
        if best_kernel is None or not is_valid_cuda_kernel(best_kernel):
            print(f"Skipping {kernel_name} (no valid optimization)")
            continue
        
        optimized_kernels[kernel_name] = best_kernel
    
    print("\n[STEP 8] Integrating ALL optimized kernels...\n")
    
    working_code = cuda_code
    
    for name, code in optimized_kernels.items():
        print(f"\nReplacing {name}...\n")
    
        updated_code = replace_kernel_in_code(working_code, name, code)
    
        if updated_code is None:
            print(f"Failed replacing {name}")
            continue
    
        if code.strip() not in updated_code:
            print(f"Kernel {name} NOT inserted correctly")
            continue
    
        print(f"Successfully replaced {name}")
    
        working_code = updated_code
    print("\n[DEBUG] Checking if optimized kernel exists...\n")
    found = False
    for k in optimized_kernels.values():
        if k.strip() in working_code:
            found = True
            break
    
    if not found:
        print("WARNING: Optimized kernels NOT inserted!")
    else:
        print("Optimized kernels correctly inserted")
    print("\n[STEP 7] Final LLM integration repair...\n")
    
    final_prompt = f"""
    You are given a FULL CUDA application.
    
    Several kernels were already optimized.
    
    Your job:
    - FIX ONLY integration/runtime/compilation issues
    - DO NOT remove optimized kernels
    - DO NOT simplify kernels
    - DO NOT revert optimizations
    - Preserve all optimized logic
    
    STRICT:
    - Keep optimized kernels EXACTLY as-is
    - Fix launch configs
    - Fix variable mismatches
    - Fix memory issues
    - Fix dimension mismatches
    - Fix undefined variables
    - Fix block/grid launch problems
    - Fix integration only
    
    OUTPUT:
    Return ONLY corrected CUDA code.
    
    CUDA CODE:
    {working_code}
    """
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0
    )
    
    final_code = response.choices[0].message.content
    
    final_code = final_code.replace("```cpp", "")
    final_code = final_code.replace("```cuda", "")
    final_code = final_code.replace("```", "")
    
    filename = r"E:\GPU Kernel Optimization\polybenchGpu-master\polybenchGpu-master\CUDA\JACOBI2D\jacobi2D_modified.cu"
    
    save_optimized_file(final_code, 0, filename)
    
    print("\nFinal repaired integrated CUDA file saved")
            
main()
