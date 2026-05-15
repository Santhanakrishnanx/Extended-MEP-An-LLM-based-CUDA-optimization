import os
import subprocess
import re
from Kernel_Fusion.fusion_preprocess import replace_kernel_in_code

def validate_full_integration(kernel_code, original_code, kernel_name):
    temp_code = replace_kernel_in_code(original_code, kernel_name, kernel_code)

    if temp_code is None:
        return None

    success = validate_full_program(temp_code)

    if not success:
        return None

    run = subprocess.run(["temp_full.exe"], capture_output=True, text=True)
    
    output = run.stdout
    
    match = re.search(r"GPU Time in seconds:\s*([0-9.eE+-]+)", output)
    if match:
        return float(match.group(1))

    return None

def validate_full_program(code):

    BENCHMARK_DIR = r"E:\GPU Kernel Optimization\polybenchGpu-master\polybenchGpu-master\CUDA\JACOBI2D"

    temp_cu = os.path.join(BENCHMARK_DIR, "temp_full.cu")
    temp_exe = os.path.join(BENCHMARK_DIR, "temp_full.exe")

    with open(temp_cu, "w", encoding="utf-8") as f:
        f.write(code)

    compile_cmd = [
        "nvcc",
        "temp_full.cu",
        "../../common/polybench.c",
        "-I../../common",
        "-O2",
        "-o",
        "temp_full.exe"
    ]

    compile = subprocess.run(
        compile_cmd,
        cwd=BENCHMARK_DIR,   
        capture_output=True,
        text=True
    )

    if compile.returncode != 0:
        print("Compile failed")
        print(compile.stderr)
        return False

    run = subprocess.run(
        ["temp_full.exe"],
        cwd=BENCHMARK_DIR,   
        capture_output=True,
        text=True
    )

    print(run.stdout)

    if "Non-Matching CPU-GPU Outputs" in run.stdout:
        match = re.search(r":\s*(\d+)", run.stdout)

        if match and int(match.group(1)) != 0:
            return False

    return True
