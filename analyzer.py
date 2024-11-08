from tree_sitter import Language, Parser, Node
from cparser import *
import os
import tree_sitter_c
import warnings

# Variable class
target_function = 'stackOverflow'
# Path to C file
filename = 'simpleTB/simple1.c'

class BasicBlock:
    def __init__(self, leader):
        self.leader = leader  # The instruction that starts this block
        self.instructions = []  # All instructions in this block
        self.declarations = []  # Variable declarations for this block
        self.changed_variables = set()  # Variables assigned/modified in this block

    def add_instruction(self, instruction):
        self.instructions.append(instruction)
        if isinstance(instruction, Declaration):
            self.declarations.append(instruction.variable.name)
        # Track assignments
        if isinstance(instruction, Assignment):
            self.changed_variables.add(instruction.variable.name)

    def __repr__(self):
        return (f"""BasicBlock( 
                instructions={self.instructions}, 
                declarations={self.declarations}, 
                changed_variables={self.changed_variables})
                """)

class LoopAbstract:
    def __init__(self, instructions, cond):
        self.instructions = instructions
        self.cond = cond
    def __repr__(self):
        return f"Loop(cond={self.cond}, instructions={self.instructions})"
class BasicBlockAnalyzer:
    def __init__(self, instructions):
        self.instructions = instructions
        self.blocks = []
        self.leaders = set()

    def identify_leaders(self):
        """Identify leaders in the instruction list based on the rules."""
        if self.instructions:
            # First instruction is always a leader
            self.leaders.add(0)

        for i, instruction in enumerate(self.instructions):
            if isinstance(instruction, Control):
                if instruction.control_type == 'ForLoop':
                    # Init statement in a for loop marks a leader for the loop
                    self.leaders.add(i)
                    # The statement following the loop is also a leader
                    if i + 1 < len(self.instructions):
                        self.leaders.add(i + 1)
            elif hasattr(instruction, 'jump_target'):
                # Target of a conditional or unconditional jump is a leader
                target_index = instruction.jump_target
                self.leaders.add(target_index)
                # The instruction following the jump is also a leader
                if i + 1 < len(self.instructions):
                    self.leaders.add(i + 1)

    def generate_blocks(self):
        """Generate basic blocks from the instructions using identified leaders."""
        self.identify_leaders()
        leaders = sorted(self.leaders)
        current_block = None

        for i, instruction in enumerate(self.instructions):
            # Check if this instruction is a leader
            if i in leaders:
                if current_block is not None:
                    self.blocks.append(current_block)
                # Start a new basic block at the current leader
                current_block = BasicBlock(leader=instruction)

            # Add instruction to the current block
            if isinstance(instruction,Control):
                if instruction.control_type == 'ForLoop':
                    self.blocks[-1].add_instruction(instruction.initialization)
                    inside_loop = [instruction.body, instruction.update]
                    loop = LoopAbstract(inside_loop,instruction.condition)
                    current_block.add_instruction(loop)
            else:
                current_block.add_instruction(instruction)

        # Add the last block if any
        if current_block is not None:
            self.blocks.append(current_block)

    def analyze_blocks(self):
        """Run the analysis to generate and print basic blocks with declarations."""
        self.generate_blocks()
        for block in self.blocks:
            print(block)

class Analyzer:
    def __init__(self, instructions):
        self.instructions = instructions
        self.state = {}  # Map from variable names to their abstract values

    def analyze(self):
        # Start analyzing the main instruction list
        for instruction in self.instructions:
            self.analyze_instruction(instruction)

    def analyze_instruction(self, instruction):
        # Print current instruction and state
        #print(instruction)
        print("Current state:", self.state)

        if isinstance(instruction, Declaration):
            # Handle variable declaration and initial state assignment
            var_name = instruction.variable.name
            value = instruction.variable.value
            size = instruction.variable.size
            a_size = 1
            if size is not None:
                for s in size:
                    a_size *= int(s)
                interval = [0, a_size - 1]
            else:
                interval = [value, value]  # Treat single value as [value, value]
            self.state[var_name] = [value, interval]

        elif isinstance(instruction, Assignment):
            # Handle assignment updates to existing variables
            var_name = instruction.variable.name
            value = instruction.value
            # Update state for the variable if already declared
            if var_name in self.state:
                # For now, just set the value directly; add more processing if needed
                self.state[var_name][0] = value
            else:
                print(f"Warning: {var_name} assigned before declaration.")

        elif isinstance(instruction, Control):
            # Process control structures (like loops and conditionals)
            if instruction.initialization:
                init = self.analyze_instruction(instruction.initialization)
            if instruction.condition:
                self.analyze_instruction(instruction.condition)
            if instruction.update:
                self.analyze_instruction(instruction.update)

            # Analyze instructions within the control structure's body
            for instr in instruction.body:
                self.analyze_instruction(instr)
            for instr in instruction.else_body:
                self.analyze_instruction(instr)

        elif isinstance(instruction, Binary):
            # Analyze left and right operands of binary operations recursively
            left = self.analyze_instruction(instruction.left)
            right = self.analyze_instruction(instruction.right)
            # Example: Returning result of binary expression if needed
            return self.evaluate_binary_expression(left, right, instruction.operation)

        elif isinstance(instruction, Unary):
            # Analyze the operand of a unary operation
            operand = self.analyze_instruction(instruction.operand)
            # Example: Returning result of unary expression if needed
            return self.evaluate_unary_expression(operand, instruction.operation)

        elif isinstance(instruction, FunctionCall):
            # Optionally handle function calls if relevant to state tracking
            print(f"Function call to {instruction.name} with arguments {instruction.arguments}")

        # Add additional cases for other instruction types as needed
        elif isinstance(instruction,str):
            return instruction
        else:
            print(f"Cannot analyze {instruction.instruction_type}")

    def evaluate_binary_expression(self, left, right, operation):
        # Handle binary operation and return the result for further analysis
        # Placeholder for actual logic (add, subtract, etc.)
        if operation == '+':
            return left + right
        elif operation == '-':
            return left - right
        # Add cases for other operations as needed
        return None

    def evaluate_unary_expression(self, operand, operation):
        # Handle unary operation and return the result
        # Placeholder for actual logic (e.g., negation)
        if operation == '-':
            return -operand
        return operand

    def check_oob(self, array_name, index):
        # Check if the index is within the bounds of the array
        array_bounds = self.state.get(array_name)
        if array_bounds:
            lower_bound, upper_bound = array_bounds[1]
            if not (lower_bound <= index <= upper_bound):
                print(f"Potential OOB access: {array_name}[{index}]")




# Example usage
if __name__ == "__main__":
    
    
    with open(filename, 'r') as file:
        c_code = file.read()
    
    # Initialize analyzer and parse code
    c_parser = setup_tree_sitter()
    c_parser = CParser(c_parser)
    c_parser.build(c_code,target_function)
    c_parser.print_args()
    #c_parser.print_instructions()
    #anal = Analyzer(c_parser.instructions)
    #anal.analyze()
    analyzer = BasicBlockAnalyzer(c_parser.instructions)
    analyzer.analyze_blocks()

