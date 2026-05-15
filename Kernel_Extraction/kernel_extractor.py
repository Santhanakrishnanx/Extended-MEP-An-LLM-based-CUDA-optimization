import re

def extract_all_kernels(cuda_code):
    pattern = r"__global__\s+void\s+(\w+)\s*\("
    matches = list(re.finditer(pattern, cuda_code))

    kernels = []

    for match in matches:
        name = match.group(1)
        start_index = match.start()

        brace_start = cuda_code.find("{", start_index)
        if brace_start == -1:
            continue

        brace_count = 0
        end_index = None

        for i in range(brace_start, len(cuda_code)):
            if cuda_code[i] == "{":
                brace_count += 1
            elif cuda_code[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_index = i + 1
                    break

        if end_index:
            kernel_code = cuda_code[start_index:end_index]
            kernels.append({
                "name": name,
                "code": kernel_code
            })

    return kernels

def extract_complete_kernels(text):
    import re

    text = text.replace("```cpp", "")
    text = text.replace("```cuda", "")
    text = text.replace("```c++", "")
    text = text.replace("```", "")

    kernels = []
    pattern = r"__global__\s+void\s+[A-Za-z_]\w*\s*\("

    matches = list(re.finditer(pattern, text))

    for match in matches:
        start = match.start()

        brace_start = text.find("{", start)

        if brace_start == -1:
            continue

        brace_count = 0
        end = None

        for i in range(brace_start, len(text)):

            if text[i] == "{":
                brace_count += 1

            elif text[i] == "}":
                brace_count -= 1

                if brace_count == 0:
                    end = i + 1
                    break

        if end:
            kernel = text[start:end]

            if "__global__" not in kernel:
                continue

            if "threadIdx" not in kernel:
                continue

            kernels.append(kernel)

    return kernels

def extract_signature(kernel_code):
    import re
    match = re.search(r"__global__\s+void\s+(\w+)\s*\((.*?)\)", kernel_code, re.DOTALL)
    if not match:
        return None
    return match.group(2).strip()

def extract_sig_info(sig):
    params = [p.strip() for p in sig.split(",") if p.strip()]
    return len(params)

def parse_kernel_signature(kernel_code):
    pattern = r"__global__\s+void\s+(\w+)\s*\((.*?)\)"
    match = re.search(pattern, kernel_code, re.DOTALL)

    if not match:
        return None

    name = match.group(1)
    params = match.group(2)

    param_list = []
    for p in params.split(","):
        p = p.strip()
        if p:
            tokens = p.split()
            param_name = tokens[-1]
            param_type = " ".join(tokens[:-1])
            param_list.append((param_type, param_name))

    return {
        "name": name,
        "params": param_list
    }