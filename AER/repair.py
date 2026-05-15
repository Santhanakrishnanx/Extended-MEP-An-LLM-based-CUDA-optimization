from LLM_API.openai_client import client

def repair_code_with_llm(code, error_message):
    prompt = f"""
Fix the CUDA code.

STRICT:
- Do not change logic unnecessarily
- Fix compilation/runtime errors only
- Output ONLY corrected CUDA code

Error:
{error_message}

Code:
{code}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    code = response.choices[0].message.content

    code = code.replace("```cpp", "")
    code = code.replace("```", "")
    
    return code