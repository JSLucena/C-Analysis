import subprocess
import os
import re
from enum import Enum
from typing import Optional, List, Dict, Set
from dataclasses import dataclass
from z3 import *
import networkx as nx
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
        self.cycles = None

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
        alloca_match = re.match(r"(%\w+) = alloca (.+),", line)
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
        self.ranges = {block_id: {} for block_id in blocks}
        self.array_accesses = {block_id: [] for block_id in blocks}
        self.loop_bounds = {block_id: [] for block_id in blocks}
        self.arrays_declarations = {}
        self.cycles = None

    def reaching_definitions_analysis(self):
        # Initialize worklist with all blocks
        worklist = list(self.blocks.keys())
        worklist.sort()
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
                        return
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
                """
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
                """



            # Update the block's constants
            if out_consts != self.constants[block_id]:
                self.constants[block_id] = out_consts
                # Add successors to the worklist if constants changed
                worklist.extend(block.successors)


    def get_array_declarations(self):
        worklist = list(self.blocks.keys())
        for block_id in worklist:
            block = self.blocks[block_id]
            for instr in block.instructions:
                if isinstance(instr,Alloca):
                    var = instr.dest
                    match = re.match(r"\[(\d+) x (\w+)\]", instr.op_type)
                    if match:
                        size = int(match.group(1))
                        type_ = match.group(2)
                        self.arrays_declarations[var] = [size,type_]

    def get_array_assignments(self):
        worklist = list(self.blocks.keys())
        for block_id in worklist:
            block = self.blocks[block_id]
            for instr in block.instructions:
                if isinstance(instr, GetElementPtr) and instr.base_ptr in self.arrays_declarations:
                     # Assume indices[1] is the index of the element being accessed
                    index = instr.indices[1]
                    if block_id not in self.array_accesses:
                        self.array_accesses[block_id] = []
                    self.array_accesses[block_id].append([instr.base_ptr, index])

    def get_array_index(self):
        for block_id, block in self.blocks.items():
            for instr in block.instructions:
                if isinstance(instr, GetElementPtr):
                    ret =self.find_origin(block_id,instr.indices[-1])
                    for index, arr in enumerate(self.array_accesses[block_id]):
                        if arr[0] == instr.base_ptr:
                            self.array_accesses[block_id][index][1] = ret

    def get_loop_bound(self):
        for block_id, block in self.blocks.items():
            for instr in block.instructions:
                if isinstance(instr, Branch) and instr.condition != None:
                    ret =self.find_origin(block_id,instr.condition)
                    cond = self.find_comparison(block_id,instr.condition)
                    true = self.analyze_condition(block_id,cond,True,ret)
                    false = self.analyze_condition(block_id,cond,False,ret)
                    trueblock = int(instr.true_block)
                    falseblock = int(instr.false_block)
                    self.ranges[trueblock][ret] = true
                    self.ranges[falseblock][ret] = false

                    

    def analyze_condition(self,block_id,cond,bool,var):
        #operation = cond.operation
        #dest = cond.dest
        #operand1 =cond.operand1
        operand2 = cond.operand2
        op_type = cond.op_type
        try:
            operand2 = int(operand2)
        except:
            print("fuck")
        init_val = self.constants[block_id][var]
        if op_type == 'slt':
            return [init_val,operand2-1] if bool else [operand2,operand2]
    

    def find_comparison(self,block_id,var_name):
        block = self.blocks[block_id]
        for instr in reversed(block.instructions):
            if isinstance(instr,Comparison) and var_name == instr.dest:
                return instr

    def analyze_array_access(self):
        for block_id, accesses in self.array_accesses.items():
            for acc in accesses:
                array = self.arrays_declarations[acc[0]]
                values = self.ranges[block_id][acc[1]]
                if values[0] < 0 or values[1] > array[0]:
                    return "stack overflow"
                else:
                    return "ok"

    def find_origin(self, block_id, var_name):
        block = self.blocks[block_id]
        for instr in reversed(block.instructions):
            if hasattr(instr, "dest") and instr.dest == var_name:
                if isinstance(instr, Load):
                    # If the variable is loaded from a memory location, trace back the source
                    return self.find_origin(block_id, instr.src)
                elif isinstance(instr, SignExtend):
                    # If the variable is a sign-extended value, trace back the original source
                    return self.find_origin(block_id, instr.src)
                elif isinstance(instr, MathOperation) or isinstance(instr,Comparison):
                    # If the variable is the result of a math operation, trace back the operands
                    op1 = instr.operand1
                    op2 = instr.operand2
                    #if op1 in self.constants[block_id]:
                        # If the first operand is a constant, use the constant value
                    #    return self.constants[block_id][op1]
                    #else:
                        # If the first operand is a variable, trace back its origin
                    return self.find_origin(block_id, op1)
                elif isinstance(instr, Alloca):
                    # If the variable is an alloca'd value, return the variable name
                    return var_name
                else:
                    # For other instructions, return the variable name
                    return var_name

        # If the variable is not found in the current block, check the predecessors
        for pred in self.blocks[block_id].predecessors:
            origin = self.find_origin(pred, var_name)
            if origin is not None:
                return origin

        return None

    def detect_cycles(self):
        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes and edges based on the block successors
        for label, block in self.blocks.items():
            G.add_node(label)
            for successor in block.successors:
                G.add_edge(label, successor)

        # Detect cycles
        
        try:
            cycles = list(nx.find_cycle(G, orientation='original'))
            cycle = []
            for c in cycles:
                cycle.append(c[0])
            self.cycles = cycle
        except nx.NetworkXNoCycle:
            print("No cycles found.")

    

    def display_results(self):
        print("Reaching Definitions Analysis Results:")
        for block_id, defs in self.reaching_definitions.items():
            print(f"Block {block_id}: {defs}")
        
        print("\nConstant Propagation Analysis Results:")
        for block_id, consts in self.constants.items():
            print(f"Block {block_id}: {consts}")
        
        print("\nArray declarations found:")
        for name, data  in self.arrays_declarations.items():
            print(f"array {name}, size {data[0]}, type {data[1]}")

        print("\nArray acesses found:")
        for block, arrays  in self.array_accesses.items():
            for array in arrays:
                print(f"Block {block}, array {array[0]}, index {array[1]}")

        print("Cycles detected:", self.cycles)
        for block_id, ranges in self.ranges.items():
            print(f"Analyzed ranges {block_id}: {ranges}")

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


dfa.get_array_declarations()
dfa.get_array_assignments()
dfa.detect_cycles()
dfa.get_array_index()
dfa.get_loop_bound()
ret = dfa.analyze_array_access()
# Display results
dfa.display_results()

print(ret)

#errors = dfa.check_oob_with_z3()

#if errors:
#    print("Out-of-bounds errors detected:")
#    for error in errors:
#        print(f"Block ID: {error[0]}, Array: {error[1]}, Index Variable: {error[2]}")
#else:
#    print("No out-of-bounds errors detected.")