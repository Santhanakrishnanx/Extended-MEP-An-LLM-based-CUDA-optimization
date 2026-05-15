import os
import subprocess
import tempfile
from Compilation.execution import run_executable
from AER.repair import repair_code_with_llm
from execution import extract_time

def compile_cuda(cu_path, exe_path):
    compile_cmd = ["nvcc", cu_path, "-O2", "-o", exe_path]

    try:
        proc = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if proc.returncode != 0:
            print("\nCOMPILATION ERROR:\n")
            print(proc.stderr)
        
        return proc.returncode == 0, proc.stderr or proc.stdout

    except Exception as e:
        return False, str(e)
    
def compile_and_run(mep_code, save_debug=False):
    result = {
        "compile_success": False,
        "run_success": False,
        "compile_error": None,
        "runtime_error": None,
        "runtime_output": None,
        "execution_time": None
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        cu_path = os.path.join(tmpdir, "temp.cu")
        exe_path = os.path.join(tmpdir, "temp.exe")

        with open(cu_path, "w", encoding="utf-8") as f:
            f.write(mep_code)
        success, err = compile_cuda(cu_path, exe_path)

        if not success:
            result["compile_error"] = err
            return result

        result["compile_success"] = True
        success, out, err = run_executable(exe_path)

        if not success:
            result["runtime_error"] = err
            return result

        result["run_success"] = True
        result["runtime_output"] = out
        result["execution_time"] = extract_time(out)

        if save_debug:
            debug_path = os.path.join(os.getcwd(), "debug_last_run.cu")
            with open(debug_path, "w") as f:
                f.write(mep_code)

    return result

def compile_run_with_repair(mep_code, max_retries=3):
    current_code = mep_code

    for attempt in range(max_retries):
        result = compile_and_run(current_code)

        if result["compile_success"] and result["run_success"]:
            return result, current_code

        error_msg = result["compile_error"] or result["runtime_error"]
        print("\n[ERROR MESSAGE]:")
        print("\n[ATTEMPT FAILED]")
        print("Compile Success:", result["compile_success"])
        print("Run Success:", result["run_success"])
        print("Error:", error_msg)
        if not error_msg:
            
            error_msg = "Runtime crash (possible illegal memory access)"
            
        repaired = repair_code_with_llm(current_code, error_msg)
        bad_patterns = [
            "const int 16",
            "const int 32",
            "const int 8",
            "const int =",
            "__global__ void void",
        ]
        
        if (
            "__global__ void" not in repaired
            or any(p in repaired for p in bad_patterns)
        ):
            print("Bad repair rejected")
            continue
        else:
            current_code = repaired

    return result, current_code
