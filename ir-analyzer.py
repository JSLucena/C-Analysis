import subprocess
import os
import re

def generate_and_analyze_ir(c_file_path, analysis_function, target_function):
    # Define the output path for the LLVM IR file
    ir_file_path = c_file_path.replace(".c", ".ll")

    try:
        # Generate LLVM IR using clang
        subprocess.run(
            ["clang", "-S", "-emit-llvm", c_file_path, "-o", ir_file_path],
            check=True
        )
        print(f"Generated IR file: {ir_file_path}")

        # Perform analysis on the specific function in the IR
        analysis_function(ir_file_path, target_function)

    finally:
        # Clean up: delete the IR file after analysis
        if os.path.exists(ir_file_path):
            os.remove(ir_file_path)
            print(f"Deleted IR file: {ir_file_path}")

def analyze_ir(ir_file_path, target_function):
    with open(ir_file_path, 'r') as f:
        ir_content = f.read()

        # Updated regex pattern to capture only the target function's IR
        pattern = rf"define.*@{re.escape(target_function)}\(.*?{{\n.*?}}"
        match = re.search(pattern, ir_content, re.DOTALL)

        if match:
            function_ir = match.group(0)
            print(f"Function '{target_function}' IR:")
            print(function_ir)
        else:
            print(f"Function '{target_function}' not found in IR.")

# Usage
c_file = "simpleTB/simple1.c"
generate_and_analyze_ir(c_file, analyze_ir, target_function="stackOverflow")
