from tree_sitter import Language, Parser, Node
import os
import tree_sitter_c
import warnings
# Variable class
target_function = 'allocate_memory'
# Path to C file
filename = 'simpleTB/mid6.c'


types = ['primitive_type']
literals = ['char_literal', 'number_literal', 'string_literal']
parse_not_required = ['{' , '}', '(', ')', ';' , 'for', 'comment']
class Variable:
    def __init__(self, name, vtype, size=None, value=None):
        self.name = name
        self.type = vtype
        self.size = size  # Only for arrays
        self.value = value  # Store current value if assigned

    def __repr__(self):
        return f"Variable(name={self.name}, type={self.type}, size={self.size}, value={self.value})"

class Parameter(Variable):
    def __init__(self, name, var_type):
        super().__init__(name, var_type)
    
    def __repr__(self):
        return f"Parameter({self.name}: {self.var_type})"

class Pointer(Variable):
    def __init__(self, name, pointee_type, value=None):
        super().__init__(name, pointee_type, value=value)
        self.is_pointer = True  # Mark this as a pointer

    def __repr__(self):
        return f"Pointer(name={self.name}, type={self.type}, value={self.value})"
# Base Instruction class
class Instruction:
    def __init__(self, instruction_type):
        self.instruction_type = instruction_type

    def __repr__(self):
        return f"{self.instruction_type}"


# Declaration class
class Declaration(Instruction):
    def __init__(self, variable):
        super().__init__("Declaration")
        self.variable = variable

    def __repr__(self):
        return f"{self.instruction_type}({self.variable})"


# Extended Control class to handle branches
class Control(Instruction):
    def __init__(self, control_type, initialization=None, condition=None, update=None):
        super().__init__("Control")
        self.control_type = control_type
        self.initialization = initialization  # For-loop initialization or None
        self.condition = condition  # Condition for if/loop
        self.update = update  # For-loop update or None
        self.body = []  # Instructions inside the main body
        self.else_body = []  # Instructions inside the else branch, if present

    def add_instruction(self, instruction, branch="if"):
        if branch == "if":
            self.body.append(instruction)
        elif branch == "else":
            self.else_body.append(instruction)

    def __repr__(self):
        return (f"""{self.control_type}(
                init->{self.initialization}, 
                cond->{self.condition}, 
                update->{self.update}, 
                body->{self.body}, 
                else->{self.else_body}) """)

class Binary(Instruction):
    def __init__(self, operation, left, right):
        super().__init__("Binary")
        self.operation = operation
        self.left = left
        self.right = right

    def __repr__(self):
        return f"{self.instruction_type}({self.left} {self.operation} {self.right})"
    
class Unary(Instruction):
    def __init__(self, operand,operation, is_prefix):
        super().__init__("Unary")
        self.operation = operation
        self.operand = operand
        self.is_prefix = is_prefix

    def __repr__(self):
        prefix = self.operation if self.is_prefix else ""
        postfix = "" if self.is_prefix else self.operation
        return f"{prefix}{self.operand}{postfix}"
    
class PointerRef(Instruction):
    def __init__(self, operation,operand):
        super().__init__("PointerRef")
        self.operation = operation
        self.operand = operand

    def __repr__(self):
        return f"{self.instruction_type}({self.operation}{self.operand})"
    
class Assignment(Instruction):
    def __init__(self, variable, value, operation="=", index=None):
        super().__init__("Assignment")
        self.variable = variable
        self.value = value
        self.operation = operation  # Assignment operator (e.g., "=", "+=", "*=", etc.)
        self.index = index # Optional for arrays

    def __repr__(self):
        return f"{self.instruction_type}({self.variable}[{self.index}] {self.operation} {self.value} )"

class Cast(Instruction):
    def __init__(self, value, cast):
        super().__init__("Cast")
        self.value = value
        self.cast = cast

    def __repr__(self):
        return f"{self.instruction_type}({self.cast}, {self.value})"

# FunctionCall class to represent a function call
class FunctionCall(Instruction):
    def __init__(self, name, arguments):
        super().__init__("FunctionCall")
        self.name = name  # Name of the function being called
        self.arguments = arguments  # List of arguments passed to the function

    def __repr__(self):
        args = ', '.join(map(str, self.arguments))
        return f"""{self.instruction_type}(
        func->{self.name} 
        args->{args})"""

# Analyzer class
class CParser:
    def __init__(self, parser):
        self.parser = parser
        self.variables = {}  # Dictionary of variables
        self.instructions = []  # List of ordered instructions

    def build(self, code, function_name):
        tree = self.parser.parse(bytes(code, "utf8"))
        
        self.process_function(tree.root_node, code, function_name)


    def process_function(self,node,code, function_name):
        if node.type == "function_definition":

            declarator = node.children[1]
            name = declarator.children[0].text.decode('ascii')
            if name == function_name:
                self.function_name = name
                self.return_type = node.children[0].text.decode('ascii')
                body = node.children[2]
                self.parameters = self.parse_parameters(declarator, code)
                self.process_nodes(body,code)
            
                
        else:  
            for child in node.children:
                self.process_function(child,code, function_name)


    def parse_parameters(self, node, code):
        """
        Parses the parameter list of a function definition.
        Returns a list of Variable instances representing each parameter.
        """
        parameters = []
        
        # Find the parameter list node
        param_list_node = next((child for child in node.children if child.type == 'parameter_list'), None)
        
        if param_list_node:
            for param in param_list_node.children:
                if param.type == 'parameter_declaration':
                    # Get the type and name of each parameter
                    type_node = next((child for child in param.children if child.type in types), None)
                    identifier_node = next((child for child in param.children if child.type == 'identifier'), None)
                    
                    if type_node and identifier_node:
                        param_type = code[type_node.start_byte:type_node.end_byte]
                        param_name = code[identifier_node.start_byte:identifier_node.end_byte]
                        
                        # Create a Variable instance for each parameter
                        parameter = Variable(param_name, param_type)
                        parameters.append(parameter)
                        
        return parameters


    def process_nodes(self, node, code):
        for child in node.children:
            inst = self.process_node(child,code)
            if inst != None:
                if isinstance(inst,list):
                    for thing in inst:
                        self.instructions.append(thing)
                else:
                    self.instructions.append(inst)
    def process_node(self,node,code):
        if node.type == 'declaration':
                declarations = self.handle_declaration(node, code)
                return declarations

        elif node.type == 'for_statement':
            for_instruction = self.handle_for_statement(node, code)
            if for_instruction:
                return for_instruction

        elif node.type == 'binary_expression':
            binary_instruction = self.handle_binary(node, code)
            if binary_instruction:
                return binary_instruction

        elif node.type == 'expression_statement':
            expr = self.handle_expression_statement(node,code)
            return expr
        elif node.type == 'assignment_expression':
            expr = self.handle_assignment(node,code)
            return expr
        elif node.type == 'if_statement':
            expr = self.handle_if_statement(node,code)
            return expr
        
        elif node.type == 'break_statement' or node.type == 'continue_statement':
            control_instr = self.handle_break_continue(node)
            return control_instr
        
        elif node.type == 'while_statement':
            control_instr = self.handle_while(node, code)
            return control_instr
        
        elif node.type == 'do_statement':
            control_instr = self.handle_do_while(node, code)
            return control_instr

        elif node.type == 'identifier' or node.type == 'type_descriptor':
            return node.text.decode('ascii')
        elif node.type in literals:
            return self.extract_literal_value(node,code)
        elif node.type == 'initializer_list':
            return self.extract_initializer_list(node,code)
        elif node.type == 'call_expression':
            # Handle function call expressions (e.g., func())
            expression = self.handle_function_call(node, code)
            return expression
        elif node.type == 'init_declarator':
            expr = self.handle_init_declarator(node,code)
            return expr
        elif node.type == 'array_declarator':
            expr = self.handle_array_declarator(node,code)
            return expr
        elif node.type == 'subscript_expression':
            expr = self.handle_subscription_expression(node,code)
            return expr
        elif node.type == 'pointer_declarator':
            expr = self.handle_pointer_declarator(node,code)
            return expr
        elif node.type == 'pointer_expression':
            expr = self.handle_pointer_expression(node,code)
            return expr
        elif node.type == 'parenthesized_expression' or node.type == 'parenthesized_declarator':
            expr = node.named_children[0]
            expr = self.process_node(expr,code)
            return expr
        elif node.type == 'cast_expression':
            expr = self.handle_cast_expression(node,code)
            return expr
        elif node.type == 'sizeof_expression':
            expr = self.handle_sizeof_expression(node,code)
            return expr
        else:
            if not node.type in parse_not_required:
                print(f'Cannot parse {node.type}')
                return None

    def handle_sizeof_expression(self,node,code):
        name = node.children[0].text.decode('ascii')
        arg = node.named_children[-1]
        if arg.type != 'identifier' and arg.type != 'type_descriptor':
            arg = self.process_node(arg,code)
        else:
            arg = arg.text.decode('ascii')
        return FunctionCall(name,[arg])
    def handle_cast_expression(self,node,code):
        cast_type = node.named_children[0].text.decode('ascii')
        val = node.named_children[-1]
        if val.type == 'identifier':
            val = val.text.decode('ascii')
        else:
            val = self.process_node(val,code)
        return Cast(val,cast_type)
    def handle_pointer_expression(self,node,code):
        id = node.named_children[0]
        if id.type == 'identifier':
            id = node.named_children[0].text.decode('ascii')
        else:
            id = self.process_node(id,code)
        ptr = node.children[0].text.decode('ascii')
        
        return PointerRef(ptr,id)
    def handle_pointer_declarator(self,node,code):
        ptrs = ""
        inner_node = node.named_children[0]       
        if inner_node.type == 'identifier':
            var_name = node.text.decode('ascii')
            return (Pointer(var_name,None,None))
        else:
            inst = self.process_node(inner_node,code)
            inst.name = node.text.decode('ascii')
            if inner_node.type == 'pointer_declarator':
                ptrs += 'ptr'
                inst.type = ptrs

            
            return inst
            #print(inst)
    def handle_break_continue(self, node):
        """
        Handles `break` and `continue` statements.
        Returns a BreakContinue instruction.
        """
        # Determine the type of control instruction based on node type
        control_type = "break" if node.type == 'break_statement' else "continue"
        return Instruction(control_type)

    def handle_binary(self, node, code):
        # Check if the left and right children of the node are binary expressions
        if not node.named_children[0].type  in literals and not node.named_children[0].type == 'identifier':
            left_operand = self.process_node(node.children[0], code)
        else:
            # If the left child is not a binary expression, get its literal code
            left_operand = code[node.children[0].start_byte:node.children[0].end_byte]

        # Operator is the second child
        operation = code[node.children[1].start_byte:node.children[1].end_byte]

        # Check if the right child is a binary expression
        if not node.named_children[1].type  in literals and not node.named_children[1].type == 'identifier':
            right_operand = self.process_node(node.named_children[1], code)
        else:
            # If the right child is not a binary expression, get its literal code
            right_operand = code[node.children[2].start_byte:node.children[2].end_byte]

        # Create a Binary instruction using either subexpressions or literals
        return Binary(operation, left_operand, right_operand)


        # Modified handle_unary function to handle complex operands
    def handle_unary(self, node, code):
        """
        Process a unary expression node to create a Unary instruction.
        Supports complex operands by recursively parsing binary expressions.
        """
        # Check if the unary operator is the first or last child
        if node.children[0].type in {'++', '--'}:
            # Prefix operation
            operation = code[node.children[0].start_byte:node.children[0].end_byte]
            operand_node = node.children[1]
            is_prefix = False
        else:
            # Postfix operation
            operand_node = node.children[0]
            operation = code[node.children[1].start_byte:node.children[1].end_byte]
            is_prefix = True

        # Recursively process the operand if it's an expression
        if operand_node.type == 'binary_expression':
            operand = self.handle_binary(operand_node, code)
        elif operand_node.type == 'unary_expression':
            operand = self.handle_unary(operand_node, code)
        else:
            # Otherwise, just extract the literal operand (e.g., a single variable like 'i')
            operand = code[operand_node.start_byte:operand_node.end_byte]

        # Create and return a Unary instruction
        return Unary(operation, operand, is_prefix)

    def handle_init_declarator(self,node,code):
        inner_node = node.named_children[0]
        value = node.named_children[1]
        if not value.type in literals:
            value = self.process_node(value,code)
        if inner_node.type == 'identifier':
            var_name = inner_node.text.decode('ascii')
            return (Variable(var_name, None, None, value))
        else:
            inst = self.process_node(inner_node,code)
            if inst.type == 'array_declarator':
                value = self.extract_initializer_list(inst)
                inst.value = value
            
            return inst
            
    def handle_array_declarator(self,node,code):
        inner_node = node.named_children[0]
        if inner_node.type == 'identifier':
            size = []
            try:
                size.append(node.named_children[1].text.decode('ascii'))
            except:
                pass
            var_name = inner_node.text.decode('ascii')
            return (Variable(var_name, None, size, None))
        else:
            inst = self.process_node(inner_node,code)
            if inner_node.type == 'array_declarator': #multi dimensional arrays
                inst.size.append(node.named_children[-1].text.decode('ascii'))       
            return inst
    def handle_declaration(self, node, code):
        """
        Process a variable declaration in the syntax tree.
        Returns a list of Declaration objects for multiple variables declared in a single statement.
        """
        declarations = []
        declaration_nodes = [child for child in node.children if child.type in {'identifier', 'init_declarator', 'array_declarator', 'pointer_declarator'}]
        
        for declaration_node in declaration_nodes:
            inst = self.process_node(declaration_node,code)
            type_node = next((child for child in node.children if child.type in types), None)
            if declaration_node.type == 'identifier':
                var_name = inst
                var_type = code[type_node.start_byte:type_node.end_byte]
                var = Variable(var_name,var_type,None,None)
                declarations.append(Declaration(var))
            else:
                if isinstance(inst.type,str):
                    inst.type = code[type_node.start_byte:type_node.end_byte] + inst.type
                else:
                    inst.type = code[type_node.start_byte:type_node.end_byte]
                if declaration_node.type == 'init_declarator':
                    value = declaration_node.named_children[-1]
                    value = self.process_node(value,code)
                    if isinstance(value,Node):
                        if value.type == 'initializer_list':
                            value = self.extract_initializer_list(value,code)

                        elif value.type in literals:
                            value = self.extract_literal_value(value,code)
                    inst.value = value
                    
                declarations.append(Declaration(inst))

        return declarations


       
    def extract_literal_value(self, declarator_node, code):
        """
        Extracts literal value from the initializer node if available.
        """
        if not declarator_node.type  in literals:
            literal_node = next((child for child in declarator_node.children if child.type in literals), None)
        else:
            literal_node = declarator_node
        binary_node = next((child for child in declarator_node.children if child.type == 'binary_expression'), None)
        if literal_node:
            value_text = literal_node.text.decode()
            if literal_node.type == 'char_literal':
                return value_text.replace("'", "")
            elif literal_node.type == 'string_literal':
                return value_text[1:-1]
            return value_text
        return binary_node

    def extract_initializer_list(self, declarator_node, code):
        """
        Extracts values from an initializer list for array declarations.
        """
        if declarator_node.type == 'initializer_list':
            init_list_node = declarator_node
        else:
            init_list_node = next((child for child in declarator_node.children if child.type == 'initializer_list'), None)
        if init_list_node:
            return init_list_node.text.decode()[1:-1]  # Remove braces from the initializer list
        return None

    def handle_for_statement(self, node, code):
        init = None
        cond = None
        update = None
        # Extract initialization and condition
        for child in node.children:
            if child.type == 'declaration':
                init = self.handle_declaration(child,code)
                init = init[0]
            elif child.type == 'assignment_expression':
                init = self.process_node(child,code)
            elif child.type == 'binary_expression':
                cond = self.handle_binary(child,code)
            elif child.type == 'update_expression':
                update = self.handle_unary(child,code)

        control = Control("ForLoop", init, cond,update)



        # Parse loop body recursively and add to the control body
        for child in node.children:
            if child.type == 'compound_statement':
                for stmt in child.children:
                    inst = self.process_node(stmt,code)
                    if inst != None:
                        if not isinstance(inst,list):
                            control.add_instruction(inst)
                        else:
                            for i in inst:
                                control.add_instruction(i)
                    #control.add_instruction(self.process_node(stmt, code))

        return control

    def handle_expression_statement(self, node, code):
        """
        Processes an expression statement, which could be an assignment, binary expression, 
        unary expression, or other statement types. Returns an appropriate Instruction.
        """
        # Check the type of expression and handle accordingly
        expression = None

        if node.children[0].type == 'assignment_expression':
            # Handle assignment expressions (e.g., x = 5, x += 5)
            expression = self.handle_assignment(node.children[0], code)
            
        elif node.children[0].type == 'binary_expression':
            # Handle binary expressions (e.g., x + y, a * b)
            expression = self.handle_binary(node.children[0], code)
            
        elif node.children[0].type == 'update_expression':
            # Handle unary expressions (e.g., ++x, --y)
            expression = self.handle_unary(node.children[0], code)
            
        elif node.children[0].type == 'call_expression':
            # Handle function call expressions (e.g., func())
            expression = self.handle_function_call(node.children[0], code)

            
        else:
            print(f"Unsupported expression statement type: {node.children[0].type}")

        return expression

    def handle_function_call(self, node, code):
        """
        Parses a function call node and returns a FunctionCall object.
        """
        # First child is the function identifier
        func_name_node = node.children[0]
        func_name = code[func_name_node.start_byte:func_name_node.end_byte]

        # Second child should be the argument list
        arg_list_node = node.named_children[1]  # Assuming this is structured as "(arg1, arg2, ...)"
        arguments = []

        for arg_node in arg_list_node.named_children:
            # Check if the argument is an expression or literal and parse accordingly
            #args = 
            arguments.append(self.process_node(arg_node,code))

        # Create and return a FunctionCall instruction
        return FunctionCall(name=func_name, arguments=arguments)


    def handle_assignment(self, node, code):
        """
        Processes an assignment expression node, supporting compound assignments.
        Returns an Assignment instruction.
        """
        # Left side (variable being assigned)
        variable_node = node.children[0]
        idx = None
        rets = self.process_node(variable_node,code)
        #print(rets)
        if isinstance(rets,list):
            var = rets[0]
            subscripts = rets[1:]
            # Assignment operator (e.g., '=', '+=', '*=')
            operation = code[node.children[1].start_byte:node.children[1].end_byte]
            value_node = node.named_children[-1]
            value = self.process_node(value_node,code)
            ass = Assignment(var,value,operation,subscripts)
        else:
            var = rets
            operation = code[node.children[1].start_byte:node.children[1].end_byte]
            value_node = node.named_children[-1]
            value = self.process_node(value_node,code)
            ass = Assignment(var,value,operation,None)
        # Right side (expression being assigned)
       
        return ass

    def handle_subscription_expression(self,node,code):
        kid = node.named_children[0]
        value = node.named_children[-1]
        if value.type in literals:
            value = self.extract_literal_value(value,code)
        else:
            value = self.process_node(value,code)
        if kid.type == 'identifier':
            
            return [kid.text.decode('ascii'),value]
        else:
            inst = self.process_node(kid,code)
            inst.append(value)
            return inst
    
    # Function to parse if statements
    def handle_if_statement(self, node, code):
        """
        Parses an if statement, including optional else or else-if branches.
        Returns a Control object representing the if statement.
        """
        # Check the condition node
        condition_node = next((child for child in node.children if child.type == 'parenthesized_expression'), None)
        condition = self.process_node(condition_node.children[1],code)
        
        # Initialize Control instance for the if statement
        if_control = Control(control_type="if", condition=condition)
        
        # Process the body of the if statement
        if_body_node = next((child for child in node.children if child.type == 'compound_statement'), None)
        if if_body_node:
            for statement in if_body_node.children:
                instruction = self.process_node(statement, code)
                if instruction:
                    if_control.add_instruction(instruction, branch="if")
        
        # Process the else branch, if present
        else_node = next((child for child in node.children if child.type == 'else_clause'), None)
        if else_node:
            else_body_node = next((child for child in else_node.children if child.type == 'compound_statement'), None)
            if else_body_node:
                for statement in else_body_node.children:
                    instruction = self.process_node(statement, code)
                    if instruction:
                        if_control.add_instruction(instruction, branch="else")
        
        return if_control
    

    def handle_while(self, node, code):
        """
        Parses a `while` loop node.
        Returns a Control object representing the `while` loop.
        """
        # Extract condition from the first child (condition expression)
        condition = node.named_children[0]
        condition = self.process_node(condition.named_children[0],code)
        # Create the Control instruction for the while loop
        while_control = Control(control_type="while", condition=condition)

        # Process body of the while loop (third child is the body)
        body_node = node.children[2]
        for body_child in body_node.children:
            instruction = self.process_node(body_child, code)  # Recursive processing
            if instruction:
                while_control.add_instruction(instruction)

        return while_control

    def handle_do_while(self, node, code):
        """
        Parses a `do-while` loop node.
        Returns a Control object representing the `do-while` loop.
        """
        # Create the Control instruction for the do-while loop
        do_while_control = Control(control_type="do-while")

        # Process the body of the do-while loop (first child is the body)
        body_node = node.named_children[0]
        for body_child in body_node.children:
            instruction = self.process_node(body_child, code)  # Recursive processing
            if instruction:
                do_while_control.add_instruction(instruction)

        # Condition is the last child (condition expression after "while")
        condition = node.named_children[1]
        condition = self.process_node(condition.named_children[0],code)
        do_while_control.condition = condition

        return do_while_control


    def print_instructions(self):
        for instr in self.instructions:
            print(instr)

    def print_args(self):
        for arg in self.parameters:
            print(arg)

# Setup function for Tree-sitter
def setup_tree_sitter():
    """Initialize the Tree-sitter parser with C language support."""
    # Create Language instance
    C_LANGUAGE = Language(tree_sitter_c.language())
    parser = Parser(C_LANGUAGE)

    return parser




# Example usage
if __name__ == "__main__":
    
    
    with open(filename, 'r') as file:
        c_code = file.read()
    
    # Initialize analyzer and parse code
    parser = setup_tree_sitter()
    parser = CParser(parser)
    parser.build(c_code,target_function)
    parser.print_args()
    parser.print_instructions()
