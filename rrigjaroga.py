import parse_test


if __name__ == "__main__":
    # Path to C file
    target_function = 'simpleDeclarations'
    filename = 'parserTesting/tests.c'
    
    with open(filename, 'r') as file:
        c_code = file.read()
    
    # Initialize analyzer and parse code
    parser = setup_tree_sitter()
    parser = CParser(parser)
    parser.build(c_code,target_function)
    parser.print_args()
    parser.print_instructions()