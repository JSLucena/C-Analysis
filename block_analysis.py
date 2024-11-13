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
    def __init__(self, block_id):
        self.block_id = block_id
        self.instructions = []
        self.declarations = []
        self.control = None
        self.changed_variables = set()
        self.successors = []  # List of successor block IDs
        self.predecessors = []  # List of predecessor block IDs
        
    def add_instruction(self, instruction):
        self.instructions.append(instruction)
        if isinstance(instruction, Declaration):
            self.declarations.append(instruction.variable.name)
        if isinstance(instruction, Assignment):
            self.changed_variables.add(instruction.variable)
        if isinstance(instruction,Unary):
            if instruction.operation in ['++', '--', '+=', '-=', '*=', '/=']:
                self.changed_variables.add(instruction.operand)
            
    def add_successor(self, block_id):
        if block_id not in self.successors:
            self.successors.append(block_id)
            
    def add_predecessor(self, block_id):
        if block_id not in self.predecessors:
            self.predecessors.append(block_id)
            
    def __repr__(self):
        return (f"""BasicBlock(
    id={self.block_id},
    instructions={self.instructions},
    declarations={self.declarations},
    control={self.control},
    changed_variables={self.changed_variables},
    successors={self.successors},
    predecessors={self.predecessors}
)""")

class BasicBlockAnalyzer:
    def __init__(self, instructions):
        self.instructions = instructions
        self.blocks = {}  # Dictionary mapping block_id to BasicBlock objects
        self.current_block_id = 0

    def new_block(self):
        block = BasicBlock(self.current_block_id)
        self.blocks[self.current_block_id] = block
        self.current_block_id += 1
        return block
        
    def generate_blocks(self):
        current_block = self.new_block()
        
        for i, instruction in enumerate(self.instructions):
            if isinstance(instruction, Control):
                if instruction.control_type == 'ForLoop':
                    # Create initialization block
                    current_block.add_instruction(instruction.initialization)
                    init_block_id = current_block.block_id
                    
                    # Create condition block
                    cond_block = self.new_block()
                    current_block.add_successor(cond_block.block_id)
                    cond_block.add_predecessor(init_block_id)
                    cond_block.add_instruction(instruction.condition)
                    cond_block.control = 'Loop'
                    
                    # Create loop body block
                    body_block = self.new_block()
                    cond_block.add_successor(body_block.block_id)
                    body_block.add_predecessor(cond_block.block_id)
                    for body_instr in instruction.body:
                        body_block.add_instruction(body_instr)
                    body_block.add_instruction(instruction.update)
                    body_block.add_successor(cond_block.block_id)
                    body_block.control = 'body'
                    # Create update block
                    #update_block = self.new_block()
                    #body_block.add_successor(update_block.block_id)
                    #update_block.add_predecessor(body_block.block_id)
                    #update_block.add_instruction(instruction.update)
                    #update_block.add_successor(cond_block.block_id)
                    #cond_block.add_predecessor(update_block.block_id)
                    
                    # Create exit block
                    current_block = self.new_block()
                    cond_block.add_successor(current_block.block_id)
                    current_block.add_predecessor(cond_block.block_id)
                    
                elif instruction.control_type == 'If':
                    # Handle condition
                    current_block.add_instruction(instruction.condition)
                    cond_block_id = current_block.block_id
                    
                    # Create then block
                    then_block = self.new_block()
                    current_block.add_successor(then_block.block_id)
                    then_block.add_predecessor(cond_block_id)
                    for then_instr in instruction.body:
                        then_block.add_instruction(then_instr)
                    
                    # Create else block if it exists
                    if instruction.else_body:
                        else_block = self.new_block()
                        current_block.add_successor(else_block.block_id)
                        else_block.add_predecessor(cond_block_id)
                        for else_instr in instruction.else_body:
                            else_block.add_instruction(else_instr)
                    
                    # Create merge block
                    current_block = self.new_block()
                    then_block.add_successor(current_block.block_id)
                    current_block.add_predecessor(then_block.block_id)
                    if instruction.else_body:
                        else_block.add_successor(current_block.block_id)
                        current_block.add_predecessor(else_block.block_id)
            else:
                current_block.add_instruction(instruction)
    
    def print_blocks(self):
        """Generate and analyze the basic blocks"""
        #self.generate_blocks()
        for block_id, block in self.blocks.items():
            print(f"\nBlock {block_id}:")
            print(block)
            
    def get_block_graph(self):
        """Returns a dictionary representation of the control flow graph"""
        return {block_id: {
            'instructions': block.instructions,
            'successors': block.successors,
            'predecessors': block.predecessors
        } for block_id, block in self.blocks.items()}
    

class ArrayBoundsAnalysis:
    def __init__(self, cfg, blocks):
        self.cfg = cfg
        self.blocks = blocks
        self.array_bounds = {}  # Stores array size information
        self.block_array_accesses = {}  # Stores array accesses per block
        self.block_conditions = {}  # Stores the conditions that led to each block
        self.variables = {}  # Tracks variable values/ranges
        self.worklist = []
    def analyze_declaration(self, instruction):
        """Analyze array declarations to track their sizes"""
        if isinstance(instruction, Declaration):
            var_name = instruction.variable.name
            var_size = instruction.variable.size
            val = None
            if var_size:  # This is an array
                # Convert size to integer if it's a constant
                try:
                    size = int(var_size[0])
                    self.array_bounds[var_name] = (0, size - 1)
                except ValueError:
                    # Handle non-constant sizes if needed
                    self.array_bounds[var_name] = ('dynamic', var_size[0])
            else:
                
                if instruction.variable.type == 'int':
                    val = int(instruction.variable.value)
                    val = [val,val]
            self.variables[var_name] = [instruction.variable.type,val]

    def analyze_assignment(self, block_id, instruction):
        """Track variable values and array accesses"""
        if isinstance(instruction, Assignment):
            var_name = instruction.variable
            index = instruction.index

            if index:  # This is an array access
                # Track the array access for the current block
                if block_id not in self.block_array_accesses:
                    self.block_array_accesses[block_id] = []
                self.block_array_accesses[block_id].append({
                    'array': var_name,
                    'index': index,
                    'operation': instruction.operation
                })

                # Analyze the index expression
                index_value = self.evaluate_index(index[0])
                if var_name in self.array_bounds:
                    lower, upper = self.array_bounds[var_name]
                    if isinstance(lower, int):  # We have concrete bounds
                        if isinstance(index_value[1], int):
                            if index_value < lower or index_value > upper:
                                print(f"WARNING: Definite out-of-bounds access detected: "
                                      f"{var_name}[{index_value}] - Valid range is [{lower}, {upper}]")
                        elif isinstance(index_value, list):  # We have a range
                            min_idx, max_idx = index_value[1]
                            if max_idx > upper:
                                print(f"WARNING: Potential out-of-bounds access detected: "
                                      f"{var_name}[{min_idx}..{max_idx}] - Upper bound is {upper}")
                            if min_idx < lower:
                                print(f"WARNING: Potential out-of-bounds access detected: "
                                      f"{var_name}[{min_idx}..{max_idx}] - Lower bound is {lower}")
            else:  # Regular variable assignment
                value = self.evaluate_expression(instruction.value)
                self.variables[var_name] = value

    def analyze_control(self, block_id, instruction):
        """Analyze control flow instructions to track conditions"""
        if isinstance(instruction, Control):
            if instruction.control_type == 'ForLoop':
                # Track the loop condition
                self.block_conditions[block_id] = instruction.condition
                
                # Analyze the loop body
                for body_instr in instruction.body:
                    self.analyze_assignment(block_id, body_instr)
                    
                # Analyze the loop update
                self.analyze_assignment(block_id, instruction.update)
                
            elif instruction.control_type == 'If':
                # Track the if condition
                self.block_conditions[block_id] = instruction.condition
                
                # Analyze the then branch
                then_block_id = next(iter(instruction.body))
                self.block_conditions[then_block_id] = self.block_conditions[block_id]
                for then_instr in instruction.body[then_block_id]:
                    self.analyze_assignment(then_block_id, then_instr)
                
                # Analyze the else branch if it exists
                if instruction.else_body:
                    else_block_id = next(iter(instruction.else_body))
                    self.block_conditions[else_block_id] = self.negate_condition(self.block_conditions[block_id])
                    for else_instr in instruction.else_body[else_block_id]:
                        self.analyze_assignment(else_block_id, else_instr)

    def negate_condition(self, condition):
        """Negate a condition expression"""
        if isinstance(condition, Binary):
            if condition.operation == '<':
                return Binary('>=',condition.left, condition.right)
            elif condition.operation == '>=':
                return Binary('<' , condition.left, condition.right)
            elif condition.operation == '>':
                return Binary('<=', condition.left, condition.right)
            elif condition.operation == '<=':
                return Binary( '>', condition.left, condition.right)
            elif condition.operation == '==':
                return Binary('!=',condition.left, condition.right)
            elif condition.operation == '!=':
                return Binary('==', condition.left, condition.right)
        return Binary('!', condition)

    def evaluate_index(self, index):
        """Evaluate index expressions to determine possible values"""
        if isinstance(index, str):  # Variable
            return self.variables.get(index, ('unknown', 'unknown'))
        elif isinstance(index, (int, float)):  # Constant
            return index
        elif isinstance(index, Binary):  # Binary operation
            if index.operation == '+':
                left = self.evaluate_index(index.left)
                right = self.evaluate_index(index.right)
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    return left + right
                # Handle ranges if needed
                return ('unknown', 'unknown')
        return ('unknown', 'unknown')

    def evaluate_expression(self, expr):
        """Evaluate expressions to track variable values"""
        try:
            if "." in expr:
                expr = float(expr)
            else:
                expr = int(expr)
        except:
            expr = expr
        if isinstance(expr, (int, float)):
            return expr
        elif isinstance(expr, str):
            return self.variables.get(expr, ('unknown', 'unknown'))
        elif isinstance(expr, Binary):
            # Basic constant folding
            left = self.evaluate_expression(expr.left)
            right = self.evaluate_expression(expr.right)
            if isinstance(left, list) and isinstance(right, (int, float)):
                if expr.operation == '+':
                    return left + right
                elif expr.operation == '-':
                    return left - right
                elif expr.operation == '*':
                    return left * right
                if expr.operation == '<':
                    return left[1][1] < right
        return ('unknown', 'unknown')


    def analyze_binary(self, block_id, instruction):
        if isinstance(instruction, Binary):
            if instruction.operation in ['>', '>=', '==', '<=', '<', '!=']:
                left_value = self.evaluate_expression(instruction.left)
                right_value = self.evaluate_expression(instruction.right)

                # If the block has multiple successors, update their conditions
                if len(self.cfg[block_id]['successors']) > 1:
                    true_successor, false_successor = self.cfg[block_id]['successors']

                    # Update the condition for the true successor
                    true_range = None
                    if isinstance(left_value,list):
                        true_range = left_value[1]
                        true_range[1] += right_value
                    if isinstance(right_value,list):
                        true_range = right_value[1]
                        true_range[1] += left_value
                    self.block_conditions[true_successor] = [instruction,true_range]  # Or a more specific representation

                    # Update the condition for the false successor
                    negated_condition = self.negate_condition(instruction)
                    self.block_conditions[false_successor] = negated_condition

                result = self.evaluate_expression(instruction)
                return result


    def analyze(self):
        """Main analysis loop"""
        # Process blocks in order (can be improved with proper CFG traversal)
        self.worklist.append(0)

        while len(self.worklist) != 0:
            current = self.worklist.pop()
            current = self.blocks[current]
            self.process_block(current)
        """
        for block_id, block_data in self.cfg.items():
            print(f"\nAnalyzing Block {block_id}")
            
            for instruction in block_data['instructions']:
                # Handle declarations to track array sizes
                self.analyze_declaration(instruction)
                
                # Handle control flow instructions
                self.analyze_control(block_id, instruction)
                
                # Handle assignments and array accesses
                self.analyze_assignment(block_id, instruction)

                #handle binary assignments
                
                self.analyze_binary(block_id,instruction)
            
            print(f"Current array bounds: {self.array_bounds}")
            print(f"Current variables: {self.variables}")
            print(f"Array accesses in this block: {self.block_array_accesses.get(block_id, [])}")
            print(f"Condition for this block: {self.block_conditions.get(block_id)}")
        """
    def analyze_unary(self,block_id,instruction):
        if isinstance(instruction,Unary):
            val = self.evaluate_expression(instruction.operand)
            if instruction.operation == '++':
                val[1][1] += 1
                self.variables[instruction.operand][1] = val[1]

    def process_block(self,block):
        for instruction in block.instructions:
                # Handle declarations to track array sizes
                self.analyze_declaration(instruction)
                
                # Handle control flow instructions
                self.analyze_control(block.block_id, instruction)
                
                # Handle assignments and array accesses
                self.analyze_assignment(block.block_id, instruction)
                self.analyze_unary(block.block_id,instruction)
                #handle binary assignments
                
                result = self.analyze_binary(block.block_id,instruction)
                
        print(f"Current array bounds: {self.array_bounds}")
        print(f"Current variables: {self.variables}")
        print(f"Array accesses in this block: {self.block_array_accesses.get(block.block_id, [])}")
        print(f"Condition for this block: {self.block_conditions.get(block.block_id)}")
        if block.control == 'Loop':
            if result == True:
                self.worklist.append(block.successors[0])
            else:
                self.worklist.append(block.successors[1])  
        else:
            if len(block.successors) > 0:
                self.worklist.append(block.successors[0])

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
     # Generate basic blocks
    block_analyzer = BasicBlockAnalyzer(c_parser.instructions)
    block_analyzer.generate_blocks()
    cfg = block_analyzer.get_block_graph()
    block_analyzer.print_blocks()
    
    # Run array bounds analysis
    bounds_analyzer = ArrayBoundsAnalysis(cfg, block_analyzer.blocks)
    bounds_analyzer.analyze()