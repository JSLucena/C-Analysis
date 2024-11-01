from tree_sitter import Language, Parser
import os
import tree_sitter_c
import warnings
# Variable class

types = ['primitive_type']
literals = ['char_literal', 'number_literal', 'string_literal']
class Variable:
    def __init__(self, name, vtype, size=None, value=None):
        self.name = name
        self.type = vtype
        self.size = size  # Only for arrays
        self.value = value  # Store current value if assigned

    def __repr__(self):
        return f"Variable(name={self.name}, type={self.type}, size={self.size}, value={self.value})"


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


# Control class for loops and branches
class Control(Instruction):
    def __init__(self, control_type, initialization=None, condition=None, update=None):
        super().__init__("Control")
        self.control_type = control_type
        self.initialization = initialization  # Initial setup (e.g., int i=0)
        self.condition = condition  # Loop or branch condition (e.g., i < 10)
        self.body = []  # Instructions inside control structure

    def add_instruction(self, instruction):
        self.body.append(instruction)

    def __repr__(self):
        return f"{self.control_type}(init={self.initialization}, cond={self.condition}, body={self.body})"

class Binary(Instruction):
    def __init__(self, operation, left, right):
        super().__init__("Binary")
        self.operation = operation
        self.left = left
        self.right = right

    def __repr__(self):
        return f"{self.left} {self.operation }{self.right}"
# Analyzer class
class Analyzer:
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
                self.process_node(body,code)
            
        else:  
            for child in node.children:
                self.process_function(child,code, function_name)

    def process_node(self, node, code):
        for child in node.children:
            if child.type == 'declaration':
                self.handle_declaration(child, code)
            elif child.type == 'for_statement':
                self.handle_for_statement(child, code)
            elif child.type == 'binary_expression':
                self.handle_binary(child,code)
            else:
                print(f'Cant parse {child.type}')

        # Recursively analyze child nodes
       # if node.type == 
        #    self.process_node(child, code)

    def handle_binary(self,node,code):
        pass

    def handle_declaration(self, node, code):
        declaration_kind = node.children[1]
        value = None
        if declaration_kind.type == 'identifier':
            identifier = declaration_kind
        elif declaration_kind.type == 'init_declarator':
            #declarator_identifier = declaration_kind.children[0]
            identifier = next((child for child in declaration_kind.children if child.type == 'identifier'), None)
            value = next((child for child in declaration_kind.children if child.type in literals), None)
            if value.type == 'char_literal':
                value = value.text.decode().replace("'","")
            elif value.type == 'string_literal':
                value  = value.text.decode()[1:-1]
            else:
                value = value.text.decode()
            #value = next((child for child in node.children if child.type in literals), None)
        elif declaration_kind.type == 'array_declarator':
            identifier = next((child for child in declaration_kind.children if child.type == 'identifier'), None)
        else:
            warnings.warn(f"{declaration_kind.type} UNRESOLVED")
        type_specifier = next((child for child in node.children if child.type in types), None)
        
        if identifier and type_specifier:
            name = code[identifier.start_byte:identifier.end_byte]
            vtype = code[type_specifier.start_byte:type_specifier.end_byte]

            # Array handling
            size = None
            array_node = next((child for child in node.children if child.type == 'array_declarator'), None)
            if array_node:
                size_text = code[array_node.children[2].start_byte:array_node.children[2].end_byte]
                size = int(size_text)
            
            variable = Variable(name, vtype, size,value)
            self.variables[name] = variable
            self.instructions.append(Declaration(variable))

    def handle_for_statement(self, node, code):
        init = None
        cond = None
        update = None
        # Extract initialization and condition
        for child in node.children:
            if child.type == 'declaration':
                init = self.handle_declaration(node,code)
            elif child.type == 'binary_expression':
                cond = self.handle_binary(node,code)
            elif child.type == 'update_expression':
                update = code[child.start_byte:child.end_byte]
        control = Control("ForLoop", init, cond,update)

        # Parse loop body recursively and add to the control body
        for child in node.children:
            if child.type == 'compound_statement':
                for stmt in child.children:
                    control.add_instruction(self.process_node(stmt, code))

        self.instructions.append(control)

    def print_instructions(self):
        for instr in self.instructions:
            print(instr)


# Setup function for Tree-sitter
def setup_tree_sitter():
    """Initialize the Tree-sitter parser with C language support."""
    # Create Language instance
    C_LANGUAGE = Language(tree_sitter_c.language())
    parser = Parser(C_LANGUAGE)

    return parser




# Example usage
if __name__ == "__main__":
    # Path to C file
    filename = 'parserTesting/tests.c'
    target_function = 'declarationsWithAssignment'
    with open(filename, 'r') as file:
        c_code = file.read()

    # Initialize analyzer and parse code
    parser = setup_tree_sitter()
    analyzer = Analyzer(parser)
    analyzer.build(c_code,target_function)
    analyzer.print_instructions()
