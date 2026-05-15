import random
import re
from LLM_API.openai_client import client
from Compilation.execution import evaluate_kernel
from Kernel_Extraction.strategy_selector import get_best_strategy_from_history
from Kernel_Extraction.kernel_extractor import parse_kernel_signature
from Kernel_Extraction.kernel_extractor import extract_complete_kernels
from Kernel_Extraction.feature_extractor import extract_kernel_features
from KernelExtraction.strategy_selector import rule_based_strategy
from KernelExtraction.strategy_selector import select_strategy
from Candidate_Generation.kernel_safety import reject_bad_kernel
from Candidate_Generation.kernel_validation import is_valid_cuda_kernel
from Utils.memory_utils import update_memory
from Utils.memory_utils import get_hardware_profile



def generate_mep(kernel_code):
    sig = parse_kernel_signature(kernel_code)

    if not sig:
        raise ValueError("Failed to parse kernel signature")

    kernel_name = sig["name"]
    params = sig["params"]

    uses_x = ("threadIdx.x" in kernel_code) or ("blockIdx.x" in kernel_code)
    uses_y = ("threadIdx.y" in kernel_code) or ("blockIdx.y" in kernel_code)
    uses_z = ("threadIdx.z" in kernel_code) or ("blockIdx.z" in kernel_code)
    kernel_dim = 1
    
    if uses_y:
        kernel_dim = 2
    
    if uses_z:
        kernel_dim = 3
    index_depth = 1
    
    triple_pattern = re.search(
        r"\[[^\]]+\*[^\]]+\*[^\]]+\+",
        kernel_code
    )
    
    double_pattern = re.search(
        r"\[[^\]]+\*[^\]]+\+",
        kernel_code
    )
    
    if triple_pattern:
        index_depth = 3
    
    elif double_pattern:
        index_depth = 2

    if kernel_dim == 1:
    
        tx, ty, tz = 256, 1, 1
    
    elif kernel_dim == 2:
    
        tx, ty, tz = 16, 16, 1
    
    else:
    
        tx, ty, tz = 8, 8, 4

    if index_depth == 1:
        size_expr_val = "n"
    
    elif index_depth == 2:
        size_expr_val = "n * m"
    
    else:
        size_expr_val = "n * m * k"

    pointer_params = []
    pointer_sizes = {}

    for ptype, pname in params:
        if "*" in ptype or "*" in pname:
            pname_clean = pname.replace("*", "").strip()
            pointer_params.append(pname_clean)

    output_var = None
    
    for ptype, pname in reversed(params):
    
        pname_clean = pname.replace("*", "").strip()
    
        is_pointer = "*" in ptype or "*" in pname
    
        if not is_pointer:
            continue
    
        lower = pname_clean.lower()
    
        likely_inputs = [
            "input", "src", "image",
            "a", "b", "x", "in",
            "kernel", "filter", "mask"
        ]
    
        if lower in likely_inputs:
            continue
    
        output_var = pname_clean
        break
    
    if output_var is None and pointer_params:
        output_var = pointer_params[-1]
   
    host_alloc = []
    device_alloc = []
    memcpy_h2d = []
    kernel_args = []
    init_code = []

    for ptype, pname in params:
        pname_clean = pname.replace("*", "").strip()
        is_pointer = "*" in ptype or "*" in pname
        if is_pointer:
           
            if index_depth == 1:
                size = "n"
            
            elif index_depth == 2:
                size = "n * m"
            
            else:
                size = "n * m * k"
                            
            pointer_sizes[pname_clean] = size
            host_alloc.append(
                f"float* h_{pname_clean} = (float*)malloc(sizeof(float) * ({size}));"
            )
            
            device_alloc.append(
                f"float* d_{pname_clean}; cudaMalloc(&d_{pname_clean}, sizeof(float) * ({size}));"
            )

            if pname_clean != output_var:
                memcpy_h2d.append(
                    f"cudaMemcpy(d_{pname_clean}, h_{pname_clean}, sizeof(float)*({size}), cudaMemcpyHostToDevice);"
                )

            kernel_args.append(f"d_{pname_clean}")

        else:
            if pname_clean in ["n", "m", "k", "ni", "nj", "nk", "nl", "nm"]:

                dim_map = {
                    "n": "n",
                    "m": "m",
                    "k": "k",
                    "ni": "n",
                    "nj": "m",
                    "nk": "k",
                    "nl": "k",
                    "nm": "m"
                }
            
                value = dim_map[pname_clean]
            
            elif "float" in ptype:
                value = "1.0f"
            
            elif "double" in ptype:
                value = "1.0"
            
            else:
                value = "1"
           
            is_sweep_dim = (
                pname_clean in ["i", "j", "k"] and
                index_depth >= 3
            )
            
            if is_sweep_dim:
                continue
            host_alloc.append(f"{ptype} h_{pname_clean} = {value};")
            kernel_args.append(f"h_{pname_clean}")

    for ptype, pname in params:
        pname_clean = pname.replace("*", "").strip()
    
        if "*" in ptype or "*" in pname:
            if pname_clean == output_var:
                continue
    
            if index_depth == 1:
                size = "n"
            
            elif index_depth == 2:
                size = "n * m"
            
            else:
                size = "n * m * k"
    
            init_code.append(f"""
                    for (int i = 0; i < {size}; i++){{
                        h_{pname_clean}[i] = (float)(i % 100) / 10.0f;
                    }}
            """)

    kernel_arg_str = ", ".join(kernel_args)
   
    if kernel_dim == 1:
        block_x = tx
        block_y = 1
    
    else:
        block_x = tx
        block_y = ty
    if kernel_dim == 1:
    
        grid_config = f"""
        dim3 blocks(
            (n + {block_x} - 1) / {block_x}
        );
        """
    
    elif kernel_dim == 2:

        if "i * n + j" in kernel_code or "(i * n)" in kernel_code:
    
            grid_config = f"""
            dim3 blocks(
                (n + {block_x} - 1) / {block_x},
                (n + {block_y} - 1) / {block_y}
            );
            """
    
        else:
    
            grid_config = f"""
            dim3 blocks(
                (m + {block_x} - 1) / {block_x},
                (n + {block_y} - 1) / {block_y}
            );
            """
    
    else:
    
        grid_config = f"""
        dim3 blocks(
            (k + {block_x} - 1) / {block_x},
            (m + {block_y} - 1) / {block_y},
            n
        );
        """
    if "int i" in kernel_code and index_depth >= 3:

            dynamic_args = []
        
            for ptype, pname in params:
        
                pname_clean = pname.replace("*", "").strip()
        
                is_pointer = "*" in ptype or "*" in pname
        
                if pname_clean == "i":
                    dynamic_args.append("sweep_i")
        
                elif is_pointer:
                    dynamic_args.append(f"d_{pname_clean}")
        
                else:
                    dynamic_args.append(f"h_{pname_clean}")
        
            arg_string = ", ".join(dynamic_args)
        
            launch_code = f"""
                for(int sweep_i = 1; sweep_i < n-1; sweep_i++){{
                    {kernel_name}<<<blocks, threads>>>(
                        {arg_string}
                    );
                }}
            """
    else:
            shared_mem_size = "0"

            if "extern __shared__" in kernel_code:
            
                if kernel_dim == 2:
            
                    shared_mem_size = f"""
                    ({tx} + 2) * ({ty} + 2) * sizeof(float)
                    """
            
                elif kernel_dim == 1:
            
                    shared_mem_size = f"""
                    ({tx} + 2) * sizeof(float)
                    """
            
            launch_code = f"""
                {kernel_name}<<<blocks, threads, {shared_mem_size}>>>({kernel_arg_str});
            """
                
    mep_code = f"""
#include <iostream>
#include <cuda_runtime.h>

{kernel_code}

int main() {{
    int n = 2048;
    int m = 256;
    int k = 64;
    size_t size = sizeof(float) * ({size_expr_val});

    {" ".join(host_alloc)}

    {"".join(init_code)}

    {" ".join(device_alloc)}

    {" ".join(memcpy_h2d)}

    dim3 threads({tx}, {ty}, {tz});
    {grid_config}

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total = 0.0f;
    for (int i = 0; i < 30; i++) {{
        cudaMemset(d_{output_var}, 0, sizeof(float) * ({size_expr_val}));
        cudaEventRecord(start);
        
        {launch_code}
        
        cudaError_t err = cudaDeviceSynchronize();

        if (err != cudaSuccess) {{
            std::cout << "CUDA_ERROR: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }}

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total += ms;
    }}

    for (int rep = 0; rep < 5; rep++) {{
        cudaMemset(d_{output_var}, 0, sizeof(float) * ({size_expr_val}));
        {launch_code}
    }}
    cudaDeviceSynchronize();

    std::cout << "TIME: " << total/30 << "\\n";

    // Copy result back
    cudaMemcpy(h_{output_var},
           d_{output_var},
           sizeof(float)*({pointer_sizes[output_var]}),
           cudaMemcpyDeviceToHost);

    // Compute checksum
    double checksum = 0.0;
    double l2 = 0.0;
    
    for (int i = 0; i < ({size_expr_val}); i++) {{
        checksum += h_{output_var}[i];
        l2 += h_{output_var}[i] * h_{output_var}[i];
    }}
    
    std::cout << "CHECKSUM: " << checksum << "\\n";
    std::cout << "L2NORM: " << l2 << "\\n";
    
    return 0;
}}
"""

    return mep_code

def generate_candidates(kernel_code, num_candidates=3, strategy=None, context=None, hardware=None):
    candidates = []

    strategy_instruction = ""
    
    if strategy == "tiling":
        strategy_instruction = """
        IMPORTANT:
        - AVOID shared memory for multi-kernel pipelines
        - Prefer register blocking instead of shared memory
        - Each thread should compute multiple output elements
        - Improve arithmetic intensity
    
        DO NOT use shared memory unless explicitly required.
        """
    
    elif strategy == "memory":
        strategy_instruction = """
        Optimize memory access:
        - use shared memory tiling
        - improve coalesced access
        - reduce redundant global loads
        - use __syncthreads()
        - maximize data reuse
        """
    
    elif strategy == "unrolling":
        strategy_instruction = """
        Apply loop unrolling ONLY if loop is compute-heavy.
        Avoid increasing register pressure.
        """
    for i in range(num_candidates):
        print(f"\nGenerating candidate {i+1}...\n")
        
        prompt = f"""
        PROGRAM CONTEXT:
        {context}
        
        HARD CONSTRAINTS:
        - shared memory per block MUST be < {hardware.get("shared_mem_per_block")}
        - threads per block MUST be <= {hardware.get("max_threads_per_block")}
        - ensure threads_per_block is multiple of warp size ({hardware.get("warp_size")})
        - avoid exceeding shared memory per block
        - prefer occupancy-friendly configurations
        
        Optimize the following CUDA kernel using the given strategy.
        
        STRATEGY TO APPLY:
        {strategy_instruction}
        
        KERNEL:
        {kernel_code}
        
        Generate ONE optimized CUDA kernel.
        MANDATORY:
        - Threads may compute multiple output elements ONLY if it improves locality or arithmetic intensity.
        - Reuse loaded values in registers
        - Reduce global memory reads
        - Use coalesced memory access
        - Choose between:
            - register reuse
            - shared-memory reuse
            - loop restructuring
            based on workload characteristics.
        - DO NOT return naive implementation
        STRICT:
        - Do not create duplicate registers containing identical values
        - Do not create temporary variables unless reused multiple times
        - Avoid excessive register pressure
        - Prefer occupancy-friendly optimizations
        - Avoid large per-thread arrays
        - Avoid register arrays larger than 8 elements
        - Avoid redundant arithmetic regrouping
        - DO NOT change arithmetic expression structure
        - DO NOT combine coefficients
        - DO NOT reorder floating-point accumulation
        - DO NOT algebraically simplify expressions
        - Preserve exact numerical computation order
        - Preserve exact stencil computation semantics
        PIPELINE CONTEXT:
        - This kernel is part of a multi-kernel pipeline
        - Avoid heavy shared memory usage
        - Avoid increasing register pressure
        - Optimization must not degrade downstream kernels
        ADVANCED OPTIMIZATION OPTION:
        
        ADVANCED OPTIMIZATION OPTION:

        If kernel is part of a multi-stage pipeline AND intermediate results are written to global memory and read again:
        
        - You MAY fuse computations to eliminate intermediate global memory
        - You MUST preserve correctness
        - You MUST NOT hardcode variable names
        - You MUST infer relationships from inputs/outputs
        
        You MUST consider kernel fusion if intermediate matrices are written and re-read.
        Prefer eliminating global memory for E and F.
                
        Goal:
        - Reduce global memory traffic
        - Avoid storing intermediate matrices
        
        Example pattern:
        A → intermediate → final
        → fuse into single computation if possible
        
        STRICT REQUIREMENTS:
        - Keep EXACT same function signature
        - Keep parameter names EXACTLY SAME
        - Preserve correctness
        
        THREAD SAFETY:
        - MUST include boundary checks:
            if (row < n && col < n) OR equivalent
        - Each output element must be computed exactly once
        
        MEMORY RULES:
        - DO NOT use: extern __shared__
        - Avoid unnecessary shared memory for compute kernels
        - For stencil/Jacobi kernels, shared memory tiling is preferred
        - Avoid redundant global memory access
        - Use shared memory ONLY if blockDim == TILE_SIZE
        - Otherwise prefer register optimization
        - Ensure global memory accesses are coalesced across threads
        - Avoid strided access like B[k * nj + j]

        CRITICAL OPTIMIZATION REQUIREMENT:
        - You MUST improve memory access pattern
        - Ensure coalesced global memory access across threads
        - Avoid strided access like B[k * nj + j]
        CRITICAL FIX:
        - threadIdx.x MUST map to contiguous memory
        - DO NOT use column-wise access for global memory
        
        IMPORTANT:
        - Reorganize computation if needed
        - Prefer access pattern where threads read contiguous memory
        - Avoid unused variables
        - Minimize register pressure
        
        DO NOT:
        - Only apply loop unrolling
        - Only do minor arithmetic optimizations
                
        MEMORY ACCESS RULES:
        - Prefer contiguous global memory access across neighboring threads
        - Reduce redundant global memory loads
        - Reuse loaded values through registers when beneficial
        - Preserve original indexing semantics
        - Do not change tensor dimensionality
        - Do not assume matrices are square
        - Do not assume row-major or column-major unless evident from kernel
        - Preserve correctness for arbitrary tensor indexing
        
        TILING RULES (CRITICAL):
        - DO NOT use:
            for (int k = 0; k < TILE_SIZE; k++)
        - MUST compute valid range dynamically using:
            remaining = dimension - tile * TILE_SIZE
        - MUST handle boundary tiles correctly
        SPECIAL CASE:
        If kernel is a stencil/Jacobi/heat kernel:
        
        - Prefer shared memory tile caching
        - Load halo cells
        - Avoid excessive boundary-condition branching
        - Minimize warp divergence
        - Avoid unnecessary corner halo loads for 5-point stencils
        - Prefer cooperative loading with fewer conditionals
        - Reuse neighboring values
        - Minimize redundant global memory loads
        - Use 2D shared memory tiles
        - Use __syncthreads()
        - Avoid bank conflicts
        - Prefer stencil-style shared memory reuse
        - Threads should collaboratively load neighboring cells
        
        PERFORMANCE GOAL:
        - Minimize global memory reads
        - Use coalesced access
        - Reuse data when possible
        - Prefer shared memory reuse when beneficial
        - Avoid redundant global memory loads
        
        CRITICAL:
        - Kernel MUST be COMPLETE
        - Kernel MUST start with __global__ void
        - Kernel MUST end with closing brace
        - No truncation
        - No explanation
        - Use float2 or float4 vectorized loads if memory alignment allows
        - Ensure contiguous access across threads (threadIdx.x)
        """

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000   
        )

        text = response.choices[0].message.content
        kernels = extract_complete_kernels(text)

        if not kernels:
            print("❌ No valid kernel extracted")
            continue

        k = kernels[0]
        
        print("\n----- Generated Candidate -----\n")
        print(k)
        print("\n------------------------------\n")

        if k.count("{") != k.count("}"):
            print("Dropped (brace mismatch)")
            continue

        candidates.append(k)

    return candidates

def optimization_loop(initial_kernel, baseline_time, baseline_output, baseline_checksum, program_context, D=3, N=2):
    global global_strategy_memory
    best_time = baseline_time
    best_kernel = initial_kernel
    if best_kernel is None:
        print("No valid optimized kernel found")
        return None, None
    
    print(f"[BASELINE]: {best_time} ms")
    
    strategy_memory = {
        "tiling": [],
        "unrolling": [],
        "memory": []
    }
    
    strategies = ["tiling", "unrolling", "memory"]

    for d in range(D):
        print(f"\n=== ITERATION {d+1} ===")
        
        features = extract_kernel_features(best_kernel)
        features["kernel_type"] = program_context.get("kernel_type")
        
        rule_strategy = rule_based_strategy(features)
        history_strategy = get_best_strategy_from_history(features)
        
        if history_strategy:
            chosen_strategy = history_strategy
        else:
            chosen_strategy = select_strategy(features, d, D)
        
        
        print(f"[STRATEGY]: {chosen_strategy}")
        
        hardware = get_hardware_profile()

        candidates = generate_candidates(
            best_kernel,
            N,
            chosen_strategy,
            context=program_context,
            hardware=hardware
        )
        
        improved = False
        
        for i, cand in enumerate(candidates):
            print(f"\nCandidate {i+1}")
            if reject_bad_kernel(cand):
                print("Rejected (invalid pattern)")
                continue
            if not is_valid_cuda_kernel(cand):
                print("Invalid kernel (brace mismatch)")
                print(cand)
                continue   
            t, out = evaluate_kernel(cand, initial_kernel)
        
            if t is None:
                print("Rejected (runtime or FE fail)")
                continue
        
            print(f"Time: {t} ms")
            adjusted_time = t
            
            if t is not None:
                features_cand = extract_kernel_features(cand)
                global_strategy_memory[chosen_strategy].append({
                    "features": features_cand,
                    "time": t
                })
                strategy_memory[chosen_strategy].append(t)
           
            if adjusted_time < best_time:
                best_time = adjusted_time
                best_kernel = cand
                improved = True
            
                print(f"New best kernel accepted")
                print(f"Time: {t:.6f} ms")
            update_memory({
                "kernel_type": features_cand.get("kernel_type"),
                "memory_pattern": "strided" if features_cand["has_strided_access"] else "coalesced",
                "num_loops": features_cand["num_loops"],
                "is_2d": features_cand["is_2d"],
                "uses_shared_memory": features_cand["uses_shared_memory"],
                "optimization": chosen_strategy,
                "mep_time": t,
                "mep_speedup": baseline_time / t
            })
                                        
        if not improved:
            print("Converged early")
            continue

        print(f"[Best so far]: {best_time} ms")
        print("\n[STRATEGY MEMORY]")
        for s in strategies:
            if strategy_memory[s]:
                avg = sum(strategy_memory[s]) / len(strategy_memory[s])
                print(f"{s}: avg={avg:.4f}, runs={len(strategy_memory[s])}")
            else:
                print(f"{s}: no data")
    return best_kernel, best_time
    