from LLM_API.openai_client import client

def refine_with_llm(cuda_code, kernels):
    kernel_names = [k["name"] for k in kernels]

    prompt = f"""
From the following CUDA kernels, return the MOST compute-intensive hotspot kernel name.

Rules:
- Output ONLY one name
- Choose from this list: {kernel_names}

CUDA code:
{cuda_code}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    output = response.choices[0].message.content.strip()

    for k in kernels:
        if k["name"] == output:
            return k

    return None
