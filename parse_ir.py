from llvmlite import binding

# Initialize LLVM
binding.initialize()
binding.initialize_native_target()
binding.initialize_native_asmprinter()

# Read the IR file
with open('example.ll', 'r') as f:
    llvm_ir = f.read()

# Parse the IR
llvm_module = binding.parse_assembly(llvm_ir)
llvm_module.verify()  # Ensure the module is well-formed

# Print out functions and their basic blocks
for function in llvm_module.functions:
    print(f"Function: {function.name}")
    for block in function.blocks:
        print(f"  Block: {block.name}")
        for instr in block.instructions:
            print(f"    Instruction: {instr}")