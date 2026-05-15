def save_optimized_file(code,baseline_checksum,filename="optimized_program.cu"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"Optimized file saved as {filename}")