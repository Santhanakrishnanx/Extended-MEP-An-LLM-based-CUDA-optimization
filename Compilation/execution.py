import subprocess
import re
from compiler import compile_run_with_repair
from compiler import compile_and_run
from Candidate_Generation.candidate_generator import generate_mep
from Validation.equivalence import extract_checksum

def run_executable(exe_path):
    try:
        proc = subprocess.run(
            [exe_path],
            capture_output=True,
            text=True,
            timeout=60
        )

        print("RAW OUTPUT:", proc.stdout)
        print("STDERR:", proc.stderr)

        if proc.returncode != 0:
            error_msg = proc.stderr if proc.stderr else proc.stdout
            if not error_msg:
                error_msg = "Unknown runtime failure (likely CUDA crash)"
            return False, proc.stdout, error_msg
        
        if "CUDA_ERROR" in proc.stdout:
            return False, proc.stdout, proc.stdout

        return True, proc.stdout, proc.stderr

    except Exception as e:
        return False, None, str(e)
    
def extract_time(output):
    match = re.search(r"TIME:\s*([0-9.eE+-]+)", output)
    return float(match.group(1)) if match else None

def evaluate_kernel(candidate_kernel, baseline_kernel):
    mep_code = generate_mep(candidate_kernel)

    print("\n====== MEP FOR CANDIDATE ======\n")
    print(mep_code)
    print("\n================================\n")
    result, fixed_code = compile_run_with_repair(mep_code)

    if (not result["compile_success"]) or (not result["run_success"]):
        print("Hard failure")
        return None, None
    
    if not result["run_success"]:
        print("Compile failed but execution may still be usable")


    
    if "i * n + j" in candidate_kernel:
    
        test_shapes = [
            (128,128,1),
            (256,256,1),
            (512,512,1),
        ]
    
    else:
    
        test_shapes = [
            (128,128,64),
            (256,128,64),
            (128,256,128),
        ]
        
    all_times = []
    
    for (ni_val, nj_val, nk_val) in test_shapes:
    
        modified_code = fixed_code
    
        modified_code = re.sub(
            r"int\s+n\s*=\s*\d+;",
            f"int n = {ni_val};",
            modified_code
        )
        
        modified_code = re.sub(
            r"int\s+m\s*=\s*\d+;",
            f"int m = {nj_val};",
            modified_code
        )
        
        modified_code = re.sub(
            r"int\s+k\s*=\s*\d+;",
            f"int k = {nk_val};",
            modified_code
        )

        modified_code = re.sub(
            r"sizeof\(float\)\s*\*\s*\(n \* m \* k\)",
            f"sizeof(float) * ({ni_val} * {nj_val} * {nk_val})",
            modified_code
        )
        run_result = compile_and_run(modified_code)
    
        if not run_result["run_success"]:
            print("Run failed for shape:", ni_val, nj_val)
            return None, None
    
        output = run_result["runtime_output"]
                
        baseline_mep = generate_mep(baseline_kernel)
        
        baseline_mep = re.sub(
            r"int\s+n\s*=\s*\d+;",
            f"int n = {ni_val};",
            baseline_mep
        )
        
        baseline_mep = re.sub(
            r"int\s+m\s*=\s*\d+;",
            f"int m = {nj_val};",
            baseline_mep
        )
        
        baseline_mep = re.sub(
            r"int\s+k\s*=\s*\d+;",
            f"int k = {nk_val};",
            baseline_mep
        )
        
        baseline_mep = re.sub(
            r"sizeof\(float\)\s*\*\s*\(n \* m \* k\)",
            f"sizeof(float) * ({ni_val} * {nj_val} * {nk_val})",
            baseline_mep
        )
        
        baseline_result = compile_and_run(baseline_mep)
        
        if not baseline_result["run_success"]:
            print("Baseline failed")
            return None, None
        
        baseline_checksum = extract_checksum(
            baseline_result["runtime_output"]
        )
        
        candidate_checksum = extract_checksum(output)
        
        if candidate_checksum is None or baseline_checksum is None:
            print("Missing checksum")
            return None, None
        
        rel_error = abs(candidate_checksum - baseline_checksum) / max(abs(baseline_checksum), 1e-8)
        
        if rel_error > 5e-2:
            print("FE mismatch")
            print("Baseline:", baseline_checksum)
            print("Candidate:", candidate_checksum)
            print("Shape:", ni_val, nj_val)
            return None, None

        t = extract_time(output)
        if t is None:
            print("Time extraction failed")
            return None, None
    
        all_times.append(t)
    
    times = sorted(all_times)
    
    median_time = sorted(all_times)[len(all_times)//2]
    
    if max(all_times) > 3 * median_time:
        print("Timing outlier ignored")
        all_times.remove(max(all_times))
        
    if len(times) >= 3:
        times = times[1:-1]
    
    avg_time = sum(times) / len(times)
    return avg_time, output

def evaluate_performance(mep_code, runs=10, trim=2):
  
    result, fixed_code = compile_run_with_repair(mep_code)

    if not result["run_success"]:
        return None

    exe_time = []

    for i in range(runs + 1):

        if i == 0:
            compile_and_run(fixed_code)
            continue
    
        run_result = compile_and_run(fixed_code)
    
        if not run_result["run_success"]:
            continue   # do NOT kill evaluation
    
        t = extract_time(run_result["runtime_output"])
    
        if t is None:
            continue
    
        exe_time.append(t)

    exe_time.sort()
    
    if len(exe_time) >= 5:
        exe_time = exe_time[1:-2]   
    
    avg = sum(exe_time) / len(exe_time)
    return avg
    