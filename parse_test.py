from tree_sitter import Language, Parser
import os
import tree_sitter_c

def setup_tree_sitter():
    """Initialize the Tree-sitter parser with C language support."""
    # Path to your compiled .so file
    
    # Create Language instance
    C_LANGUAGE = Language(tree_sitter_c.language())
    
    # Initialize parser
    parser = Parser(C_LANGUAGE)
    
    return parser

def parse_c_code(parser, code):
    """Parse C code and return the syntax tree."""
    tree = parser.parse(bytes(code, "utf8"))
    return tree

def print_node(node, code, level=0):
    """Print a node and its children with indentation."""
    # Get the text for the current node
    node_text = code[node.start_byte:node.end_byte]
    print("  " * level + f"{node.type}: '{node_text}'")
    
    # Recursively print children
    for child in node.children:
        print_node(child, code, level + 1)

def find_functions(node, code):
    """Find all function declarations in the code."""
    functions = []
    
    if node.type == 'function_definition':
        # Find the function declarator
        declarator = next((child for child in node.children if child.type == 'function_declarator'), None)
        if declarator:
            # Get the function name
            identifier = next((child for child in declarator.children if child.type == 'identifier'), None)
            if identifier:
                function_name = code[identifier.start_byte:identifier.end_byte]
                functions.append({
                    'name': function_name,
                    'start_line': identifier.start_point[0],
                    'end_line': node.end_point[0]
                })
    
    # Recursively search children
    for child in node.children:
        functions.extend(find_functions(child, code))
    
    return functions

def analyze_c_code(code):
    """Analyze C code and print various insights."""
    parser = setup_tree_sitter()
    tree = parse_c_code(parser, code)
    
    print("Full Syntax Tree:")
    print_node(tree.root_node, code)
    
    print("\nFunctions found:")
    functions = find_functions(tree.root_node, code)
    for func in functions:
        print(f"Function: {func['name']} (lines {func['start_line'] + 1}-{func['end_line'] + 1})")

# Example usage
if __name__ == "__main__":
    example_code = """
    #include <stdio.h>
    
    int add(int a, int b) {
        return a + b;
    }
    
    int main() {
        int result = add(5, 3);
        printf("Result: %d\\n", result);
        return 0;
    }
    """
    
    analyze_c_code(example_code)