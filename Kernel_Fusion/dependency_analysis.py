import re
from Kernel_Extraction.kernel_extractor import parse_kernel_signature

def detect_kernel_chain(kernels):
    chain = []

    for k in kernels:
        sig = parse_kernel_signature(k["code"])
        if not sig:
            continue

        ptrs = []
        for ptype, pname in sig["params"]:
            if "*" in ptype or "*" in pname:
                pname_clean = pname.replace("*", "").strip()
                ptrs.append(pname_clean)

        if len(ptrs) < 2:
            continue

        output_var = ptrs[-1]
        input_vars = ptrs[:-1]

        chain.append({
            "name": k["name"],
            "inputs": input_vars,
            "output": output_var
        })

    return chain

def find_fusable_patterns(chain):
    patterns = []

    for i in range(len(chain)):
        for j in range(len(chain)):
            if i == j:
                continue

            if chain[i]["output"] in chain[j]["inputs"]:
                patterns.append((chain[i], chain[j]))

    return patterns

def should_fuse(features, context):
    return (
        context["dependency"] == "multi_kernel_pipeline"
        and features["num_loops"] <= 2   
    )

def final_dependency_fix_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()

    code = code.replace("```cuda", "")
    code = code.replace("```", "")

    lines = code.split("\n")
    cleaned_lines = []

    for line in lines:
        # remove explanation lines
        if line.strip().startswith("- "):
            continue
        if "Summary of fixes" in line:
            continue

        cleaned_lines.append(line)

    code = "\n".join(cleaned_lines)

    if "#include <stdio.h>" not in code:
        code = "#include <stdio.h>\n" + code

    code = re.sub(
        r"dim3\s+block\s*\([^)]*\);",
        "dim3 block(TILE_SIZE, TILE_SIZE);",
        code
    )

    code = code.replace("DIM_THREAD_BLOCK_X", "TILE_SIZE")
    code = code.replace("DIM_THREAD_BLOCK_Y", "TILE_SIZE")

    return code
