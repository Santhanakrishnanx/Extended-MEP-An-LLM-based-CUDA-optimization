import os
import json
import subprocess
import tempfile

MEMORY_FILE = "kernel_memory.json"

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def update_memory(entry):
    memory = load_memory()
    memory.append(entry)
    save_memory(memory)


def get_hardware_profile():

    code = r'''
    #include <stdio.h>
    #include <cuda_runtime.h>

    int main() {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        printf("NAME:%s\n", prop.name);
        printf("WARP:%d\n", prop.warpSize);
        printf("MAX_THREADS:%d\n", prop.maxThreadsPerBlock);
        printf("SHARED_MEM:%zu\n", prop.sharedMemPerBlock);
        printf("SM_COUNT:%d\n", prop.multiProcessorCount);

        return 0;
    }
    '''

    with tempfile.TemporaryDirectory() as tmpdir:
        cu_path = os.path.join(tmpdir, "device_query.cu")
        exe_path = os.path.join(tmpdir, "device_query.exe")

        with open(cu_path, "w") as f:
            f.write(code)

        subprocess.run(["nvcc", cu_path, "-o", exe_path], capture_output=True)
        result = subprocess.run([exe_path], capture_output=True, text=True)
        output = result.stdout

    profile = {}

    for line in output.split("\n"):
        if "NAME:" in line:
            profile["gpu_name"] = line.split(":")[1]
        elif "WARP:" in line:
            profile["warp_size"] = int(line.split(":")[1])
        elif "MAX_THREADS:" in line:
            profile["max_threads_per_block"] = int(line.split(":")[1])
        elif "SHARED_MEM:" in line:
            profile["shared_mem_per_block"] = int(line.split(":")[1])
        elif "SM_COUNT:" in line:
            profile["multiprocessor_count"] = int(line.split(":")[1])

    return profile