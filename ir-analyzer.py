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
  """
  Analyzes the IR file and returns the IR for each occurrence of the target function.

  Args:
      ir_file_path: Path to the IR file.
      target_function: Name of the function to analyze.

  Returns:
      A dictionary where keys are function names (if multiple matches) and values are the corresponding IR strings.
  """
  with open(ir_file_path, 'r') as f:
    ir_content = f.read()

  # Use re.finditer for all matches
  pattern = r"define dso_local (\S*) @([a-zA-Z0-9_]+)\((.*?)\) #0 \{([\s\S]*?)\}"
  matches = re.finditer(pattern, ir_content, re.DOTALL)

  function_data = {}  # Dictionary to store results

  for match in matches:
    if match.group(2) == target_function:  # Check if function name matches target
      function_name = match.group(2)
      function_params = match.group(3)
      function_ir = match.group(4)
      function_data[function_name] = function_ir
      print(f"Function '{target_function}' (Occurrence):")
      print(f"  Parameters: {function_params}")
      print(f"  IR:\n{function_ir}\n")

  if not function_data:
    print(f"Function '{target_function}' not found in IR.")

  return function_data
       
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

class CallInstruction:
    def __init__(self, dest, func_name, func_type, args):
        self.dest = dest
        self.func_name = func_name
        self.func_type = func_type
        self.args = args

    def __repr__(self):
        return f"CallInstruction(dest={self.dest}, func_name={self.func_name}, func_type={self.func_type}, args={self.args})"

class Bitcast(Instruction):
    def __init__(self, dest, source_type, target_type):
        super().__init__(f"bitcast {source_type} to {target_type}", target_type)
        self.dest = dest
        self.source_type = source_type
        self.target_type = target_type


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
            #print(line)
            instruction = self.parse_instruction(line)
            if instruction:
                self.current_block.add_instruction(instruction)

        return self.blocks

    def parse_instruction(self, line):
        match = re.match(r"(%\d+) = bitcast (.*) (%.*) to (.*)", line)

        if match:
            dest_reg = match.group(1)
            source_type = match.group(3)
            target_type = match.group(4)

            return Bitcast(dest_reg, source_type, target_type)


        match = re.match(r"(%\d+) =\s+srem\s+(.*)\s+(%\d+),\s+(\d+)", line)
        if match:
            dest = match.group(1)
            operation = "srem"
            op_type = match.group(2)
            operand1 = match.group(3)
            operand2 = match.group(4)
            return MathOperation(operation, dest, operand1, operand2, op_type)
        
        match = re.match(r"(%\d+) =\s+srem\s+(.*)\s+(\d+),\s+(\d+)", line)
        if match:
            dest = match.group(1)
            operation = "srem"
            op_type = match.group(2)
            operand1 = match.group(3)
            operand2 = match.group(4)
            return MathOperation(operation, dest, operand1, operand2, op_type)
        
        match = re.match(r"(%\d+) =\s+srem\s+(.*)\s+(%\d+),\s+(%\d+)", line)
        if match:
            dest = match.group(1)
            operation = "srem"
            op_type = match.group(2)
            operand1 = match.group(3)
            operand2 = match.group(4)
            return MathOperation(operation, dest, operand1, operand2, op_type)
        
        match = re.match(r"(%\d+) =\s+srem\s+(.*)\s+(\d+),\s+(%\d+)", line)
        if match:
            dest = match.group(1)
            operation = "srem"
            op_type = match.group(2)
            operand1 = match.group(3)
            operand2 = match.group(4)
            return MathOperation(operation, dest, operand1, operand2, op_type)

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
            operation = cmp_match.group(3)
            op_type = cmp_match.group(2)
            operand1 = cmp_match.group(4)
            operand2 = cmp_match.group(5)
            return Comparison(operation, dest, operand1, operand2, op_type)
        cmp_match = re.match(r"(%\w+) = (icmp|fcmp) (.+?) (.+?) (%\w+), (\w+)", line)
        if cmp_match:
            dest = cmp_match.group(1)
            operation = cmp_match.group(3)
            op_type = cmp_match.group(2)
            src_type = cmp_match.group(4)
            operand1 = cmp_match.group(5)
            operand2 = cmp_match.group(6)
            return Comparison(operation, dest, operand1, operand2, op_type)
        
        cmp_match = re.match(r"(%\w+) = (icmp|fcmp) (.+?) (.+?) (\w+), (\w+)", line)
        if cmp_match:
            dest = cmp_match.group(1)
            operation = cmp_match.group(3)
            op_type = cmp_match.group(2)
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

        call_match = re.match(r"(?:(%\w+) = )?call (\S+) (\S+)\((.*)\)",line)
        if call_match:
            dest = call_match.group(1)  # Destination register (if any)
            func_type = call_match.group(2)  # Function type (e.g., void, i8*)
            func_name = call_match.group(3)  # Function name (e.g., @malloc)
            args_str = call_match.group(4)  # Argument list as a string

            # Split the argument string and capture type and name for each argument
            args = []
            for arg in args_str.split(", "):
                if "getelementptr inbounds" in arg:
                    gep_match = re.match(r"(.+?) (.*) \((.*)", arg)
                    arg_type = arg_match.group(1)
                    arg_value = arg_match.group(2)
                    args.append((arg_type,arg_value))
                arg_match = re.match(r"(\S+) (%?\w+) (%?\w+)", arg)
                if arg_match:
                    arg_type = arg_match.group(1)
                    arg_value = arg_match.group(3)
                    args.append((arg_type, arg_value))

            return CallInstruction(dest, func_name, func_type, args)
        
        call_match = re.match(r"(?:(%\w+) = )?call noalias (\S+) (\S+)\((.*)\)",line)
        if call_match:
            dest = call_match.group(1)  # Destination register (if any)
            func_type = call_match.group(2)  # Function type (e.g., void, i8*)
            func_name = call_match.group(3)  # Function name (e.g., @malloc)
            args_str = call_match.group(4)  # Argument list as a string

            # Split the argument string and capture type and name for each argument
            args = []
            for arg in args_str.split(", "):
                arg_match = re.match(r"(\S+) (%?\w+) (%?\w+)", arg)
                if arg_match:
                    arg_type = arg_match.group(1)
                    arg_data = arg_match.group(2)
                    arg_value = arg_match.group(3)
                    args.append((arg_type, arg_value))

            return CallInstruction(dest, func_name, func_type, args)
        
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
        self.pointers = {block_id: {} for block_id in blocks}
        self.available_expressions = {block_id: set() for block_id in blocks}
        self.live_variables = {block_id: set() for block_id in blocks}

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
                elif isinstance(instr, MathOperation):
                    op1 = instr.operand1
                    op2 = instr.operand2
                    op = instr.operation
                    try:
                        op1 = int(op1)
                    except ValueError:
                        return
                    try:
                        op2 = int(op2)
                    except ValueError:
                        return
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
                        return

                    try:
                        op2 = int(op2)
                    except ValueError:
                        return
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
            # Update the block's constants
            if out_consts != self.constants[block_id]:
                self.constants[block_id] = out_consts
                # Add successors to the worklist if constants changed
                worklist.extend(block.successors)
    def live_variables_analysis(self):
        # Initialize worklist with all blocks
        worklist = list(self.blocks.keys())
        worklist.sort(reverse=True)  # Process in reverse order for backward analysis
        
        # Iterate until worklist is empty
        while worklist:
            block_id = worklist.pop(0)
            block = self.blocks[block_id]
            
            # Start with the live variables at the block's exit (out set)
            out_live = set()
            for succ in block.successors:
                out_live.update(self.live_variables[succ])
            
            # Compute in set for the current block
            in_live = out_live.copy()
            for instr in reversed(block.instructions):
                if isinstance(instr,Store) or isinstance(instr,SignExtend):
                    # Kill variable being defined
                    if instr.dest in in_live:
                        in_live.remove(instr.dest)
                    # Add variables used
                    in_live.update(instr.src if isinstance(instr.src, list) else [instr.src])
                elif isinstance(instr, MathOperation) or isinstance(instr,Comparison):
                    if instr.dest in in_live:
                        in_live.remove(instr.dest)
                    in_live.add(instr.operand1)
                    in_live.add(instr.operand2)
                elif isinstance(instr,GetElementPtr):
                    if instr.dest in in_live:
                        in_live.remove(instr.dest)
                    in_live.add(instr.base_ptr)
                    in_live.add(instr.indices[-1])
                elif isinstance(instr, Load):
                    # Variables used in a load are live
                    in_live.add(instr.src)
                elif isinstance(instr, Branch):
                    # Condition variable is used
                    in_live.add(instr.condition)

            # Update live variables for this block
            if in_live != self.live_variables[block_id]:
                self.live_variables[block_id] = in_live
                # Add predecessors to the worklist if live variables changed
                worklist.extend(block.predecessors)
    def available_expressions_analysis(self):
        # Initialize worklist with all blocks
        worklist = list(self.blocks.keys())
        worklist.sort()
        
        # Iterate until worklist is empty
        while worklist:
            block_id = worklist.pop(0)
            block = self.blocks[block_id]
            
            # Gather available expressions from predecessors (in set)
            in_avail = set()
            for pred in block.predecessors:
                in_avail &= self.available_expressions[pred] if in_avail else self.available_expressions[pred]
            
            # Compute out set for the current block
            out_avail = in_avail.copy()
            for instr in block.instructions:
                if isinstance(instr,Store):
                    # Kill expressions involving the modified variable
                    out_avail = {expr for expr in out_avail if instr.dest not in expr}
                elif isinstance(instr, MathOperation) or isinstance(instr,Comparison):
                    # Add new expressions formed by binary operations
                    out_avail = {expr for expr in out_avail if instr.dest not in expr}
                    out_avail.add((instr.operation, instr.operand1, instr.operand2))
                elif isinstance(instr,SignExtend):
                    out_avail = {expr for expr in out_avail if instr.dest not in expr}
                    out_avail.add(('sext',instr.src))
                elif isinstance(instr,GetElementPtr):
                    out_avail = {expr for expr in out_avail if instr.dest not in expr}
                    out_avail.add(('getelemptr',instr.base_ptr, instr.indices[-1]))
            
            # Update available expressions for this block
            if out_avail != self.available_expressions[block_id]:
                self.available_expressions[block_id] = out_avail
                # Add successors to the worklist if available expressions changed
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

        for block_id, block in self.blocks.items():
            if var in self.constants[block_id]:
                init_val = self.constants[block_id][var]
        if op_type == 'slt':
            return [init_val,operand2-1] if bool else [operand2,operand2]
        elif op_type == 'sle':
            return [init_val,operand2] if bool else [operand2+1,operand2+1]
        else:
            print("FUDEU ANALYZE CONDITION")
    

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
            print(cycles)
            for c in cycles:
                cycle.append(c[0])
            self.cycles = cycle
        except nx.NetworkXNoCycle:
            print("No cycles found.")

    def get_pointers(self):
        for block_id, block in self.blocks.items():
            for instr in block.instructions:
                if isinstance(instr, Alloca):
                    if "*" in instr.op_type:
                        self.pointers[block_id][instr.dest] = [instr.op_type,-1,0] #-1 represents size if malloced, 0 is value

    def get_mallocs(self):
        for block_id, pointers in self.pointers.items():
            for name, data in pointers.items():
                for thing in self.reaching_definitions[block_id]:
                    if thing[0] == name:
                        ptr_expr = thing[1]
                        origin = ptr_expr.src
                        origin = self.find_origin(block_id,origin)
                        origin_expr = self.reaching_definitions[block_id]




    def display_results(self):
        print("Reaching Definitions Analysis Results:")
        for block_id, defs in self.reaching_definitions.items():
            print(f"Block {block_id}: {defs}")
        
        print("\nConstant Propagation Analysis Results:")
        for block_id, consts in self.constants.items():
            print(f"Block {block_id}: {consts}")
        
        print("\nArray declarations found:")
        for name, data in self.arrays_declarations.items():
            print(f"Array {name}, size {data[0]}, type {data[1]}")
        
        print("\nArray accesses found:")
        for block, arrays in self.array_accesses.items():
            for array in arrays:
                print(f"Block {block}, array {array[0]}, index {array[1]}")
        
        print("\nLive Variables Analysis Results:")
        for block_id, vars in self.live_variables.items():
            print(f"Block {block_id}: {vars}")
        
        print("\nAvailable Expressions Analysis Results:")
        for block_id, exprs in self.available_expressions.items():
            print(f"Block {block_id}: {exprs}")
        
        print("\nCycles detected:", self.cycles)
        for block_id, ranges in self.ranges.items():
            print(f"Analyzed ranges for Block {block_id}: {ranges}")



class AbstractInterpreter:
    def __init__(self, blocks):
        self.blocks = blocks
        self.ranges = {}  # Store variable ranges as {var_name: [min, max]}

    def interpret(self):
        current_block_id = 0
        while True:
            current_block = self.blocks[current_block_id]
            print(f"Interpreting Block {current_block_id}...")

            for instr in current_block.instructions:
                if isinstance(instr, Alloca):
                    # Regular expression to match the pattern
                    pattern = r"\[(\d+)\s+x\s+(\w+)\]"

                    # Match the string
                    match = re.match(pattern, instr.op_type)
                    if match:
                        size = match.group(1)
                        type_ = match.group(2)
                        vals = []
                        for i in range(int(size)):
                            vals.append([None,None])
                        self.ranges[instr.dest] = {"type": type_, "value" : vals}
                    else:
                        self.ranges[instr.dest] = {"type": instr.op_type, "value" : [None,None]}
                    # Initialize allocated variable with [0, 0]
                    
                    print(f"Alloca: {instr.dest}")
                elif isinstance(instr, Store):
                    # Update the range of the destination with the source range or constant
                    try:
                        src = int(instr.src)
                    except:
                        src = instr.src
                    if self.ranges[instr.dest]["type"] != "ptr":
                        if isinstance(src, int):
                            self.ranges[instr.dest]["value"] = [src, src]
                        elif src in self.ranges:
                            self.ranges[instr.dest] = self.ranges[src]
                        print(f"Store: {instr.dest} = {self.ranges[instr.dest]}")
                    else:
                        pointee = self.ranges[instr.dest]["pointee"]
                        index = self.ranges[instr.dest]["value"]
                        for i in range(index[0], index[1]):
                            self.ranges[pointee]["value"][i] = [src,src]
                        print(f"Store: {pointee} = {self.ranges[pointee]}")
                elif isinstance(instr, Load):
                    # Load the range of the source into the destination
                    if instr.src in self.ranges:
                        self.ranges[instr.dest] = self.ranges[instr.src]
                        print(f"Load: {instr.dest} = {self.ranges[instr.src]}")
                    else:
                        raise ValueError(f"Undefined variable {instr.src}")
                elif isinstance(instr, MathOperation):
                    try:
                        op1 = int(instr.operand1)
                        op1_range = [op1, op1]
                    except:
                        op1_range = self.ranges.get(instr.operand1, [0, 0])
                        op1_range = op1_range["value"]
                    try:
                        op2 = int(instr.operand2)
                        op2_range = [op2, op2]
                    except:
                        op2_range = self.ranges.get(instr.operand2, [0, 0])
                        op2_range = op2_range["value"]
                    res = None
                    if instr.operation == "add":
                        res = [
                            min( op1_range[0], op2_range[0], op1_range[0]  + op2_range[0]),
                            max(op1_range[1], op2_range[1], op1_range[1] + op2_range[1])
                        ]
                    elif instr.operation == "sub":
                        res = [
                            min( op1_range[0], op2_range[0], op1_range[0]  - op2_range[0]),
                            max(op1_range[1], op2_range[1], op1_range[1] - op2_range[1])
                        ]
                    elif instr.operation == "mul":
                        res = [
                            min(
                                op1_range[0] * op2_range[0],
                                op1_range[0] * op2_range[1],
                                op1_range[1] * op2_range[0],
                                op1_range[1] * op2_range[1],
                            ),
                            max(
                                op1_range[0] * op2_range[0],
                                op1_range[0] * op2_range[1],
                                op1_range[1] * op2_range[0],
                                op1_range[1] * op2_range[1],
                            ),
                        ]
                    elif instr.operation == "div":
                        # Handle division with care for zero-divisor ranges
                        if 0 in range(op2_range[0], op2_range[1] + 1):
                            raise ValueError("Division by zero encountered in range analysis")
                        res = [
                            min(
                                op1_range[0] // op2_range[0] if op2_range[0] != 0 else float('inf'),
                                op1_range[0] // op2_range[1] if op2_range[1] != 0 else float('inf'),
                                op1_range[1] // op2_range[0] if op2_range[0] != 0 else float('inf'),
                                op1_range[1] // op2_range[1] if op2_range[1] != 0 else float('inf'),
                            ),
                            max(
                                op1_range[0] // op2_range[0] if op2_range[0] != 0 else float('-inf'),
                                op1_range[0] // op2_range[1] if op2_range[1] != 0 else float('-inf'),
                                op1_range[1] // op2_range[0] if op2_range[0] != 0 else float('-inf'),
                                op1_range[1] // op2_range[1] if op2_range[1] != 0 else float('-inf'),
                            ),
                        ]
                    self.ranges[instr.dest] = {"type" : "i32", "value" : res}
                    print(f"MathOperation: {instr.dest} = {self.ranges[instr.dest]}")
                elif isinstance(instr, Comparison):
                    # Comparisons produce a Boolean-like range: [0, 0] or [1, 1]
                    try:
                        op1 = int(instr.operand1)
                        op1_range = [op1,op1]
                    except:    
                        op1_range = self.ranges.get(instr.operand1, [0, 0])
                        op1_range = op1_range["value"]
                    try:
                        op2 = int(instr.operand2)
                        op2_range = [op2,op2]
                    except:
                        op2_range = self.ranges.get(instr.operand2, [0, 0])
                        op2_range = op2_range["value"]
                    #TODO find out how to interpret ambiguous cases
                    if instr.operation == "seq":
                        result = [1, 1] if op1_range == op2_range else [0, 0]
                    elif instr.operation == "slt":
                        result = [1, 1] if op1_range[1] < op2_range[0] else [0, 0]
                    elif instr.operation == "sgt":
                        result = [1, 1] if op1_range[0] > op2_range[1] else [0, 0]
                    else:
                        raise ValueError(f"Unknown comparison operation: {instr.operation}")
                    self.ranges[instr.dest] = {"type": "i32", "value" : result}
                    print(f"Comparison: {instr.dest} = {self.ranges[instr.dest]}")
                elif isinstance(instr, Branch):
                    # Branch based on a condition's range
                    if instr.condition != None:
                        condition_range = self.ranges.get(instr.condition, [0, 0])["value"]
                        if condition_range[0] == 1:  # Condition is definitely true
                            current_block_id = int(instr.true_block[:1])
                        elif condition_range[1] == 0:  # Condition is definitely false
                            current_block_id = int(instr.false_block[:1])
                        else:
                            raise ValueError(
                                "Cannot determine branch target from condition range"
                            )
                        print(f"Branch: Jumping to Block {current_block_id}")
                        break  # Exit the loop to process the next block
                    else:
                        current_block_id = int(instr.true_block[1:])
                        break
                elif isinstance(instr, SignExtend):
                    self.ranges[instr.dest] = self.ranges[instr.src]
                elif isinstance(instr, GetElementPtr):
                    ptr = instr.base_ptr
                    index = instr.indices[-1]
                    index = self.ranges[index]["value"]
                    if index[0] < 0 or index[1] > len(self.ranges[ptr]["value"]):
                        return "stack overflow"
                    else:
                        self.ranges[instr.dest] = {"type" : "ptr", "value" : index, "pointee" : ptr}
                elif isinstance(instr, Return):
                    # Return ends interpretation with the range of the return value
                    ret_range = (
                        self.ranges[instr.ret_value]
                        if instr.ret_value in self.ranges
                        else None
                    )
                    print(f"Return: {ret_range}")
                    return ret_range  # End interpretation
                else:
                    print(f"CANT PARSE {instr}")
            else:
                # If no branch or return, proceed to the next successor
                if current_block.successors:
                    current_block_id = current_block.successors[0]
                else:
                    print("No more successors; ending interpretation.")
                    break


# Usage
c_file = "simpleTB/simple1.c"
target_function ="stackOverflow"
code = generate_and_analyze_ir(c_file, analyze_ir, target_function)
parser = IRParser(code[target_function])
blocks = parser.parse()

for block_label, block in blocks.items():
    print(f"Block {block_label}:")
    print(f"  Instructions: {block.instructions}")
    print(f"  Predecessors: {block.predecessors}")
    print(f"  Successors: {block.successors}")
"""
# Assuming `blocks` is already defined and populated
dfa = DataFlowAnalysis(blocks)

# Run Reaching Definitions Analysis
dfa.reaching_definitions_analysis()

# Run Constant Propagation Analysis
dfa.constant_propagation_analysis()

dfa.live_variables_analysis()
dfa.available_expressions_analysis()

dfa.get_array_declarations()
dfa.get_array_assignments()
dfa.detect_cycles()
dfa.get_array_index()
dfa.get_loop_bound()
dfa.get_pointers()
dfa.get_mallocs()
ret = dfa.analyze_array_access()
# Display results
dfa.display_results()

print(ret)
"""

interpreter = AbstractInterpreter(blocks)
ret = interpreter.interpret()
print(ret)