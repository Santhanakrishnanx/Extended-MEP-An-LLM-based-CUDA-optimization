import re
from Kernel_Extraction.kernel_extractor import extract_all_kernels
from Kernel_Extraction.kernel_extractor import extract_signature
from Kernel_Extraction.kernel_extractor import extract_sig_info
from dependency_analysis import detect_kernel_chain
from dependency_analysis import find_fusable_patterns
from fusion_llm import fuse_kernels_llm
from fusion_llm import reject_bad_fusion


def replace_kernel_in_code(original_code, kernel_name, new_kernel_code):

    old_kernel_pattern = rf"__global__\s+void\s+{kernel_name}\s*\((.*?)\)"
    old_match = re.search(old_kernel_pattern, original_code, re.DOTALL)

    new_sig = extract_signature(new_kernel_code)

    if not old_match or not new_sig:
        print("Signature extraction failed")
        return None

    old_sig = old_match.group(1)
    new_sig = new_sig
    
    if extract_sig_info(old_sig) != extract_sig_info(new_sig):
        print("Signature mismatch — rejecting replacement")
        return None

    if "__shared__" in new_kernel_code and "__shared__" not in original_code:
        print("Warning: introducing shared memory may hurt integration")


    start_pattern = rf"__global__\s+void\s+{kernel_name}\s*\("
    match = re.search(start_pattern, original_code, re.DOTALL)

    if not match:
        print("Kernel not found")
        return None

    start_index = match.start()
    brace_start = original_code.find("{", start_index)

    brace_count = 0
    end_index = brace_start

    for i in range(brace_start, len(original_code)):
        if original_code[i] == "{":
            brace_count += 1
        elif original_code[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                end_index = i + 1
                break

    new_kernel_code = "\n\n" + new_kernel_code.strip() + "\n\n"

    updated_code = (
        original_code[:start_index]
        + new_kernel_code
        + original_code[end_index:]
    )

    return updated_code

def update_kernel_launch(code, kernel_name, new_kernel_code):
  
    match = re.search(rf"__global__\s+void\s+{kernel_name}\s*\((.*?)\)", new_kernel_code, re.DOTALL)
    if not match:
        return code

    params = match.group(1)
    param_names = []

    for p in params.split(","):
        p = p.strip()
        if "*" in p:
            name = p.split()[-1].replace("*", "").strip()
            param_names.append(name)

    args = []
    for name in param_names:
        if name.startswith("d_"):
            args.append(name)
        else:
            args.append("d_" + name)

    arg_str = ", ".join(args)

    # replace kernel launch
    launch_pattern = rf"{kernel_name}\s*<<<.*?>>>\s*\(.*?\);"

    new_launch = f"{kernel_name}<<<blocks, threads>>>({arg_str});"

    code = re.sub(launch_pattern, new_launch, code)

    return code

def fusion_preprocess(cuda_code, kernels):
    print("\n[STEP 0] Fusion Preprocessing...\n")

    chain = detect_kernel_chain(kernels)

    if not chain:
        print("No dependency chain detected")
        return kernels, cuda_code

    patterns = find_fusable_patterns(chain)

    if not patterns:
        print("No fusible kernel pairs found")
        return kernels, cuda_code

    print(f"Found {len(patterns)} fusion candidates")

    for (k1, k2) in patterns:
        print(f"\nAttempting fusion: {k1['name']} + {k2['name']}")

        code1 = None
        code2 = None

        for k in kernels:
            if k["name"] == k1["name"]:
                code1 = k["code"]
            if k["name"] == k2["name"]:
                code2 = k["code"]

        if not code1 or not code2:
            continue

        fused = fuse_kernels_llm(code1, code2, k1, k2)
        
        if fused is None:
            print("Fusion returned None")
            continue
        
        if reject_bad_fusion(fused):
            continue

        cuda_code = replace_kernel_in_code(cuda_code, k1["name"], fused)
        if cuda_code is None:
            print("replace_kernel returned None — skipping fusion")
            return kernels, cuda_code
        cuda_code = update_kernel_launch(cuda_code, k1["name"], fused)
        
        print(f"Fused {k1['name']} + {k2['name']}")

        kernels = extract_all_kernels(cuda_code)

        break 

    return kernels, cuda_code

def remove_kernel(code, kernel_name):
    
    if not isinstance(code, str):
        print("ERROR: code is not string in remove_kernel")
        print("Type:", type(code))
        return code   

    pattern = rf"__global__\s+void\s+{kernel_name}\s*\(.*?\)\s*\{{.*?\}}"

    try:
        return re.sub(pattern, "", code, flags=re.DOTALL)
    except Exception as e:
        print("Regex failed:", e)
        return code