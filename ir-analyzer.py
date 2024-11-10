import subprocess
import os
import re
from enum import Enum
from typing import Optional, List, Dict, Set
from dataclasses import dataclass
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
        code = analysis_function(ir_file_path, target_function)

    finally:
        # Clean up: delete the IR file after analysis
        if os.path.exists(ir_file_path):
            os.remove(ir_file_path)
            print(f"Deleted IR file: {ir_file_path}")
        return code

def analyze_ir(ir_file_path, target_function):
    with open(ir_file_path, 'r') as f:
        ir_content = f.read()

        # Updated regex pattern to capture only the target function's IR
        pattern = rf"define.*@{re.escape(target_function)}\(.*?{{\n.*?}}"
        match = re.search(pattern, ir_content, re.DOTALL)

        if match:
            function_ir = match.group(0)
            print(f"Function '{target_function}' IR:")
            #print(function_ir)
            
        else:
            print(f"Function '{target_function}' not found in IR.")
    return function_ir
       
# Define instruction classes for common LLVM IR instructions

class Instruction:
    def __init__(self, raw_text, op_type=None):
        self.raw_text = raw_text
        self.op_type = op_type

    def __repr__(self):
        return f"{self.__class__.__name__}({self.raw_text}, type={self.op_type})"

class Load(Instruction):
    def __init__(self, dest, src, op_type):
        super().__init__(f"load {dest} from {src}", op_type)
        self.dest = dest
        self.src = src

class Store(Instruction):
    def __init__(self, src, dest, op_type):
        super().__init__(f"store {src} to {dest}", op_type)
        self.src = src
        self.dest = dest

class Branch(Instruction):
    def __init__(self, condition, true_block, false_block=None):
        super().__init__(f"branch {condition} ? {true_block} : {false_block}")
        self.condition = condition
        self.true_block = true_block
        self.false_block = false_block

class MathOperation(Instruction):
    def __init__(self, operation, dest, operand1, operand2, op_type):
        super().__init__(f"{operation} {dest} = {operand1} op {operand2}", op_type)
        self.operation = operation
        self.dest = dest
        self.operand1 = operand1
        self.operand2 = operand2

class Comparison(Instruction):
    def __init__(self, operation, dest, operand1, operand2, op_type):
        super().__init__(f"{operation} {dest} = {operand1} cmp {operand2}", op_type)
        self.operation = operation
        self.dest = dest
        self.operand1 = operand1
        self.operand2 = operand2
        self.op_type = op_type

class Alloca(Instruction):
    def __init__(self, var_name, op_type):
        super().__init__(f"alloca {var_name}", op_type)
        self.dest = var_name

class SignExtend(Instruction):
    def __init__(self, dest, src, from_type, to_type):
        super().__init__(f"sext {src} from {from_type} to {to_type}")
        self.dest = dest
        self.src = src
        self.from_type = from_type
        self.to_type = to_type

class GetElementPtr(Instruction):
    def __init__(self, dest, base_ptr, indices, element_type):
        super().__init__(f"getelementptr {base_ptr} with indices {indices}", element_type)
        self.dest = dest
        self.base_ptr = base_ptr
        self.indices = indices
        self.element_type = element_type

class Return(Instruction):
    def __init__(self, ret_type, ret_value=None):
        if ret_value:
            super().__init__(f"ret {ret_type} {ret_value}")
        else:
            super().__init__(f"ret {ret_type}")
        self.ret_type = ret_type
        self.ret_value = ret_value


class BasicBlock:
    def __init__(self, label):
        self.label = label
        self.instructions = []
        self.predecessors = []
        self.successors = []

    def add_instruction(self, instruction):
        self.instructions.append(instruction)

    def __repr__(self):
        return f"BasicBlock({self.label}, preds={self.predecessors}, succs={self.successors})"

# Define the IRParser class
class IRParser:
    def __init__(self, ir_content):
        self.ir_content = ir_content
        self.blocks = {}  # Dictionary of blocks with label as key
        self.current_block = None
        self.entry_block_initialized = False  # Track if the entry block has been created

    def parse(self):
        lines = self.ir_content.splitlines()

        # Initialize the entry block
        self.current_block = BasicBlock(0)
        self.blocks[0] = self.current_block
        self.entry_block_initialized = True

        for line in lines:
            line = line.strip()

            # Detect block labels and predecessors
            block_match = re.match(r"^(\d+):\s*;\s*preds\s*=\s*%(.+)$", line)
            if block_match:
                block_label = int(block_match.group(1))
                preds = block_match.group(2).split(", %")

                # Create or get the current block
                if block_label not in self.blocks:
                    self.blocks[block_label] = BasicBlock(block_label)
                self.current_block = self.blocks[block_label]

                # Add predecessors and update their successors
                for pred in preds:
                    pred = int(pred)
                    if pred not in self.blocks:
                        self.blocks[pred] = BasicBlock(pred)
                    self.current_block.predecessors.append(pred)
                    self.blocks[pred].successors.append(block_label)

                continue

            # Detect new blocks without predecessor information
            block_label_match = re.match(r"^(\d+):", line)
            if block_label_match:
                block_label = block_label_match.group(1)

                if block_label not in self.blocks:
                    self.blocks[block_label] = BasicBlock(block_label)
                self.current_block = self.blocks[block_label]
                continue

            # Parse instruction lines and add to the current block
            instruction = self.parse_instruction(line)
            if instruction:
                self.current_block.add_instruction(instruction)

        return self.blocks

    def parse_instruction(self, line):
        # Match Load instructions
        load_match = re.match(r"(%\w+) = load (.+), (.+) (%\w+)", line)
        if load_match:
            dest = load_match.group(1)
            op_type = load_match.group(2)
            src = load_match.group(4)
            return Load(dest, src, op_type)

        # Match Store instructions
        store_match = re.match(r"store (.+?) (%\w+), (.+?) (%\w+)", line)
        if store_match:
            op_type = store_match.group(1)
            src = store_match.group(2)
            dest = store_match.group(4)
            return Store(src, dest, op_type)
        store_match = re.match(r"store (.+?) (\w+), (.+?) (%\w+)", line)
        if store_match:
            op_type = store_match.group(1)
            src = store_match.group(2)
            dest = store_match.group(4)
            return Store(src, dest, op_type)

        # Match Conditional Branch instructions
        branch_match = re.match(r"br (i1 )?(%\w+), label %(\w+), label %(\w+)", line)
        if branch_match:
            condition = branch_match.group(2)
            true_block = branch_match.group(3)
            false_block = branch_match.group(4)
            return Branch(condition, true_block, false_block)

        # Match Unconditional Branch (br label %dest)
        unconditional_branch_match = re.match(r"br label (%\w+)", line)
        if unconditional_branch_match:
            return Branch(None, unconditional_branch_match.group(1))

        # Match Math operations (e.g., add, sub, mul, div)
        math_op_match = re.match(r"(%\w+) = (add|sub|mul|div) (.+?) (%\w+), (%\w+)", line)
        if math_op_match:
            dest = math_op_match.group(1)
            operation = math_op_match.group(2)
            op_type = math_op_match.group(3)
            operand1 = math_op_match.group(4)
            operand2 = math_op_match.group(5)
            return MathOperation(operation, dest, operand1, operand2, op_type)
        
        math_op_match = re.match(r"(%\w+) = (add|sub|mul|div) (.+?) (%\w+), (\w+)", line)
        if math_op_match:
            dest = math_op_match.group(1)
            operation = math_op_match.group(2)
            op_type = math_op_match.group(3)
            operand1 = math_op_match.group(4)
            operand2 = math_op_match.group(5)
            return MathOperation(operation, dest, operand1, operand2, op_type)
        
        math_op_match = re.match(r"(%\w+) = (add|sub|mul|div) (.+?) (\w+), (\w+)", line)
        if math_op_match:
            dest = math_op_match.group(1)
            operation = math_op_match.group(2)
            op_type = math_op_match.group(3)
            operand1 = math_op_match.group(4)
            operand2 = math_op_match.group(5)
            return MathOperation(operation, dest, operand1, operand2, op_type)
        math_op_match = re.match(r"(%\w+) = (add|sub|mul|div) (.+?) (\w+), (%\w+)", line)
        if math_op_match:
            dest = math_op_match.group(1)
            operation = math_op_match.group(2)
            op_type = math_op_match.group(3)
            operand1 = math_op_match.group(4)
            operand2 = math_op_match.group(5)
            return MathOperation(operation, dest, operand1, operand2, op_type)

        # Match Comparisons (e.g., icmp, fcmp)
        cmp_match = re.match(r"(%\w+) = (icmp|fcmp) .+? (.+?) (%\w+), (%\w+)", line)
        if cmp_match:
            dest = cmp_match.group(1)
            operation = cmp_match.group(2)
            op_type = cmp_match.group(3)
            operand1 = cmp_match.group(4)
            operand2 = cmp_match.group(5)
            return Comparison(operation, dest, operand1, operand2, op_type)
        cmp_match = re.match(r"(%\w+) = (icmp|fcmp) (.+?) (.+?) (%\w+), (\w+)", line)
        if cmp_match:
            dest = cmp_match.group(1)
            operation = cmp_match.group(2)
            op_type = cmp_match.group(3)
            src_type = cmp_match.group(4)
            operand1 = cmp_match.group(5)
            operand2 = cmp_match.group(6)
            return Comparison(operation, dest, operand1, operand2, op_type)

        # Match Alloca instructions
        alloca_match = re.match(r"(%\w+) = alloca (.+)", line)
        if alloca_match:
            var_name = alloca_match.group(1)
            op_type = alloca_match.group(2)
            return Alloca(var_name, op_type)
        
        # Match Sign Extend (sext) instructions
        sext_match = re.match(r"(%\w+) = sext (.+?) (%\w+) to (.+)", line)
        if sext_match:
            dest = sext_match.group(1)
            from_type = sext_match.group(2)
            src = sext_match.group(3)
            to_type = sext_match.group(4)
            return SignExtend(dest, src, from_type, to_type)

        # Match GetElementPtr (GEP) instructions
        gep_match = re.match(r"(%\w+) = getelementptr inbounds (.+?), (.+) (%\w+), (.+) (%\w+)", line)
        if gep_match:
            dest = gep_match.group(1)
            element_type = gep_match.group(2)
            base_ptr = gep_match.group(4)
            indices = [gep_match.group(5), gep_match.group(6)]
            return GetElementPtr(dest, base_ptr, indices, element_type)

        # Return None if no matching instruction is found
        ret_match = re.match(r"ret (void)", line)
        if ret_match:
            ret_type = ret_match.group(1)
            return Return(ret_type, 0)
        
        ret_match = re.match(r"ret (.+?) (%\w+)?", line)
        if ret_match:
            ret_type = ret_match.group(1)
            ret_value = ret_match.group(2)
            return Return(ret_type, ret_value)

        # Return None if no matching instruction is found
        return None
# Example usage




class DataFlowAnalysis:
    def __init__(self, blocks):
        self.blocks = blocks
        # To store the definitions reaching each block
        self.reaching_definitions = {block_id: set() for block_id in blocks}
        # To store constants propagated across blocks
        self.constants = {block_id: {} for block_id in blocks}

    def reaching_definitions_analysis(self):
        # Initialize worklist with all blocks
        worklist = list(self.blocks.keys())
        
        # Iterate until worklist is empty
        while worklist:
            block_id = worklist.pop(0)
            block = self.blocks[block_id]

            # Get current definitions reaching this block
            in_defs = set()
            for pred in block.predecessors:
                in_defs |= self.reaching_definitions[pred]

            # Add instructions in the block to the in_defs set
            out_defs = in_defs.copy()
            for instr in block.instructions:
                if hasattr(instr,"dest"):
                    dst = instr.dest
                    out_defs = {d for d in out_defs if d[0] != dst}
                    out_defs.add((dst,instr))
                """
                if isinstance(instr, (Alloca, Store, Load)):  # For example
                    # Define the target variable, assume it's instr.target
                    target = instr.target
                    # Remove old definitions of the target
                    out_defs = {d for d in out_defs if d[0] != target}
                    # Add the new definition
                    out_defs.add((target, instr))
                """
            # Update block's reaching definitions
            if out_defs != self.reaching_definitions[block_id]:
                self.reaching_definitions[block_id] = out_defs
                # Add successors back to the worklist if changed
                worklist.extend(block.successors)
    def constant_propagation_analysis(self):
        # Initialize worklist with all blocks
        worklist = list(self.blocks.keys())
        worklist.sort()
        # Iterate until worklist is empty
        while worklist:
            block_id = worklist.pop(0)
            block = self.blocks[block_id]
            in_consts = {}

            # Gather constant values from predecessors
            for pred in block.predecessors:
                for var, val in self.constants[pred].items():
                    if var in in_consts and in_consts[var] != val:
                        del in_consts[var]  # Different values across preds
                    else:
                        in_consts[var] = val

            # Apply constant propagation within the block
            out_consts = in_consts.copy()
            for instr in block.instructions:
                if isinstance(instr, Store):
                    # Store instruction with integer value means a constant
                    try:
                        val = int(instr.src)
                    except:
                        val = instr.src
                        #val = out_consts[val]
                    target = instr.dest
                    if val in out_consts:
                        out_consts[target] = out_consts[val]
                    else:
                        out_consts[target] = val
                elif isinstance(instr, Load):
                    # For a load, replace with constant if it exists
                    if instr.src in out_consts:
                        out_consts[instr.dest] = out_consts[instr.src]
                elif isinstance(instr, Branch) and instr.condition in out_consts:
                    # Replace branch condition if it's constant
                    instr.condition_value = out_consts[instr.condition]
                elif isinstance(instr, MathOperation):
                    op1 = instr.operand1
                    op2 = instr.operand2
                    op = instr.operation
                    try:
                        op1 = int(op1)
                    except ValueError:
                        pass
                    try:
                        op2 = int(op2)
                    except ValueError:
                        pass
                    if op1 in out_consts:
                        op1 = out_consts[op1]
                    if op2 in out_consts:
                        op2 = out_consts[op2]
                    if op == 'add':
                        out_consts[instr.dest] = op1 + op2
                    else:
                        print("ERROR MATH")
                elif isinstance(instr, Comparison):
                    op1 = instr.operand1
                    op2 = instr.operand2
                    op = instr.op_type

                    try:
                        op1 = int(op1)
                    except ValueError:
                        pass

                    try:
                        op2 = int(op2)
                    except ValueError:
                        pass

                    if op1 in out_consts:
                        op1 = out_consts[op1]
                    if op2 in out_consts:
                        op2 = out_consts[op2]

                    if isinstance(op1, int) and isinstance(op2, int):
                        if op == 'slt':
                            out_consts[instr.dest] = op1 < op2
                        elif op == 'sge':
                            out_consts[instr.dest] = op1 >= op2
                        else:
                            print("ERROR CMP")
                elif isinstance(instr,SignExtend):
                    if instr.src in out_consts:
                        out_consts[instr.dest] = out_consts[instr.src]
                    else:
                        out_consts[instr.dest] = instr.src



            # Update the block's constants
            if out_consts != self.constants[block_id]:
                self.constants[block_id] = out_consts
                # Add successors to the worklist if constants changed
                worklist.extend(block.successors)


    def display_results(self):
        print("Reaching Definitions Analysis Results:")
        for block_id, defs in self.reaching_definitions.items():
            print(f"Block {block_id}: {defs}")
        
        print("\nConstant Propagation Analysis Results:")
        for block_id, consts in self.constants.items():
            print(f"Block {block_id}: {consts}")

# Usage
c_file = "simpleTB/simple1.c"
code = generate_and_analyze_ir(c_file, analyze_ir, target_function="stackOverflow")
parser = IRParser(code)
blocks = parser.parse()



for block_label, block in blocks.items():
    print(f"Block {block_label}:")
    print(f"  Instructions: {block.instructions}")
    print(f"  Predecessors: {block.predecessors}")
    print(f"  Successors: {block.successors}")


# Assuming `blocks` is already defined and populated
dfa = DataFlowAnalysis(blocks)

# Run Reaching Definitions Analysis
dfa.reaching_definitions_analysis()

# Run Constant Propagation Analysis
dfa.constant_propagation_analysis()

# Display results
dfa.display_results()