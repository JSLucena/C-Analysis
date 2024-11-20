from cparser import setup_tree_sitter
from tree_sitter import Language, Parser
import warnings
import re

warnings.simplefilter("ignore", FutureWarning)
# Language.build_library(
#     'build/my-languages.so',
#     [
#         './tree-sitter-c'
#     ]
# )

C_LANGUAGE = Language('build/language.so', 'c')
parser = Parser()
parser.set_language(C_LANGUAGE)

# Variable class
target_function = 'stackOverflow'
# Path to C file
filename = 'simpleTB/simple1.c'

import ast
import astor

def extract_function(source_code: str, function_name: str) -> str:
    """
    Extract a function from source code, supporting various function signatures.
    
    Args:
        source_code (str): The complete source code containing the target function
        function_name (str): Name of the function to extract
    
    Returns:
        str: The extracted function code, or None if not found
    """
    # Build the pattern pieces separately to avoid f-string confusion
    function_body = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    
    pattern = (
        # Match optional return type and modifiers
        r'((?:(?:virtual|static|public|private|protected|inline|extern|const)\s+)*)?'
        # Match return type (optional)
        r'(?:[\w\d_:*&<>\s]+\s+)?'
        # Match function name
        f'{re.escape(function_name)}\\s*'
        # Match parameters
        r'\([^)]*\)\s*'
        # Match optional const modifier and trailing specifiers
        r'(?:\s*const)?\s*'
        r'(?:override|final)?\s*'
        # Match function body
        f'{function_body}'
    )
    
    # Compile pattern with verbose flag and multiline support
    compiled_pattern = re.compile(pattern, re.VERBOSE | re.MULTILINE | re.DOTALL)
    
    # Find all matches
    matches = compiled_pattern.finditer(source_code)
    
    # Extract and format the functions
    found_functions = []
    for match in matches:
        function_code = match.group(0)
        # Clean up whitespace but preserve indentation
        lines = function_code.split('\n')
        # Remove empty lines at start and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        
        # Find minimum indentation (excluding empty lines)
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines:
            min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
            # Remove minimum indentation from all lines
            lines = [line[min_indent:] if line.strip() else '' for line in lines]
        
        found_functions.append('\n'.join(lines))
    
    if not found_functions:
        return None
    
    # Return all found functions joined by newlines
    return '\n\n'.join(found_functions)

with open(filename, 'r') as file:
    code = file.read()
    print(code)

    code = extract_function(code, target_function)
    print(code)

    # c_parser = setup_tree_sitter()

#code = """
#void CWE416_Use_After_Free__malloc_free_char_01_bad()
#{
#    char * data;
#    /* Initialize data */
#    data = NULL;
#    data = (char *)malloc(100*sizeof(char));
#    if (data == NULL) {exit(-1);}
#    memset(data, 'A', 100-1);
#    data[100-1] = '\0';
#    /* POTENTIAL FLAW: Free data in the source - the bad sink attempts to use data */
#    free(data);
#    /* POTENTIAL FLAW: Use of data that may have been freed */
#    printLine(data);
#    /* POTENTIAL INCIDENTAL - Possible memory leak here if data was not freed */
#}
#
#
#"""

tree = parser.parse(bytes(code, "utf8"))
#print(tree.root_node.__dir__())

def query_loop(tree, code):
    query_string = """
    (for_statement
        initializer: (declaration) @init
        condition: (binary_expression
            left: (identifier) @loop_var
            right: (number_literal) @loop_limit
        )
        update: (update_expression) @update
    )
    """
    query = C_LANGUAGE.query(query_string)
    captures = query.captures(tree.root_node)
    
    parts = {}
    for node, capture_name in captures:
        parts[capture_name] = code[node.start_byte:node.end_byte]

    return parts


def query_array_declaration(tree, code):
    array_query_string = """
    (declaration
        type: (primitive_type) @type
        declarator: (array_declarator
            declarator: (identifier) @array_name
            size: (number_literal) @array_size
        )
    )
    """

    array_with_init_query_string = """
    (declaration
        type: (primitive_type) @type
        declarator: (init_declarator
            declarator: (array_declarator
                declarator: (identifier) @array_name
                size: (number_literal) @array_size
            )
        )
    )
    """

    query1 = C_LANGUAGE.query(array_query_string)
    query2 = C_LANGUAGE.query(array_with_init_query_string)

    captures1 = query1.captures(tree.root_node)
    captures2 = query2.captures(tree.root_node)

    arrays_info = []
    for captures in [captures1, captures2]:
        parts = {}
        for node, capture_name in captures:
            parts[capture_name] = code[node.start_byte:node.end_byte].strip()
        
        if 'array_name' in parts and 'array_size' in parts:
            arrays_info.append({
                "type": parts.get("type", ""),
                "name": parts.get("array_name", ""),
                "size": int(parts.get("array_size", 0))
            })
    
    return arrays_info

def query_malloc_calls_for_valuename(tree, code):
    malloc_query_string = """
    (declaration
        declarator: (init_declarator
            declarator: (pointer_declarator declarator: (identifier) @var_name)
            value: (cast_expression
                value: (call_expression
                    function: (identifier) @func_name
                    arguments: (argument_list (_)*) @alloc_args
                )
            )
        )
    )
    """
    query = C_LANGUAGE.query(malloc_query_string)
    captures = query.captures(tree.root_node)

    malloc_vars = []
    for node, capture_name in captures:
        if capture_name == 'var_name':
            var_name = code[node.start_byte:node.end_byte].strip()
            malloc_vars.append(var_name)
    return malloc_vars

def query_malloc_calls_forvalue(tree, code):
    malloc_query_string = """
    (declaration
        declarator: (init_declarator
            declarator: (pointer_declarator declarator: (identifier) @var_name)
            value: (cast_expression
                value: (call_expression
                    function: (identifier) @func_name
                    arguments: (argument_list
                        (number_literal) @alloc_size  ; Capture number directly
                    )
                )
            )
        )
    )
    """
    query = C_LANGUAGE.query(malloc_query_string)
    captures = query.captures(tree.root_node)

    malloc_vars = []
    for node, capture_name in captures:
        if capture_name == 'alloc_size':
            alloc_size = code[node.start_byte:node.end_byte].strip()
            malloc_vars.append(alloc_size)  # Append the actual size value
    return malloc_vars




 
def query_free_calls(tree, code):
    free_query_string = """
    (
        call_expression
            function: (identifier) @func_name
            arguments: (argument_list
                (identifier) @var_name
            )
    ) @free_call
    """

    query = C_LANGUAGE.query(free_query_string)
    captures = query.captures(tree.root_node)

    free_vars = []
    i = 0
    while i < len(captures):
        node, capture_name = captures[i]
        if capture_name == 'func_name':
            func_name = code[node.start_byte:node.end_byte].strip()
            if func_name == 'free':
                if i + 1 < len(captures):
                    next_node, next_capture_name = captures[i + 1]
                    if next_capture_name == 'var_name':
                        var_name = code[next_node.start_byte:next_node.end_byte].strip()
                        free_vars.append(var_name)
                        i += 2  
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1
    print("Free Variables Detected:", free_vars)
    return free_vars

def query_dereference(tree, code):
    dereference_query_string = """
    (expression_statement
        (assignment_expression
            left: (pointer_expression
                argument: (identifier) @var_name
            )
        )
    )
    """
    query = C_LANGUAGE.query(dereference_query_string)
    captures = query.captures(tree.root_node)

    dereference_vars = []
    for node, capture_name in captures:
        if capture_name == 'var_name':
            var_name = code[node.start_byte:node.end_byte].strip()
            dereference_vars.append(var_name)
    return dereference_vars

def query_memcpy_usage(tree, code):
    memcpy_query_string = """
    (call_expression
        function: (identifier) @func_name
        arguments: (argument_list
            (identifier) @dest
            (identifier) @src
            (sizeof_expression
                value: (parenthesized_expression (identifier) @size_of_src)
            )
        )
    )
    """
    query = C_LANGUAGE.query(memcpy_query_string)
    captures = query.captures(tree.root_node)

    memcpy_info = {}
    for node, capture_name in captures:
        text = code[node.start_byte:node.end_byte].strip()
        memcpy_info[capture_name] = text

    print("Memcpy Usage Analysis:")
    if 'func_name' in memcpy_info and memcpy_info['func_name'] == "memcpy":
        print(f"  Detected memcpy call from {memcpy_info.get('src')} to {memcpy_info.get('dest')} "
              f"with size {memcpy_info.get('size_of_src')}")

        src_array_query_string = f"""
        (declaration
            declarator: (init_declarator
                declarator: (array_declarator
                    declarator: (identifier) @src_array_name
                    size: (number_literal)? @src_array_size
                )
                value: (string_literal)? @init_value
            )
            (#eq? @src_array_name "{memcpy_info.get('src')}")
        )
        """
        
        src_query = C_LANGUAGE.query(src_array_query_string)
        src_captures = src_query.captures(tree.root_node)

        src_size = None
        for node, capture_name in src_captures:
            if capture_name == 'src_array_size':
                src_size = int(code[node.start_byte:node.end_byte].strip())
            elif capture_name == 'init_value':
                src_size = len(code[node.start_byte:node.end_byte].strip()) - 2  
        memcpy_info['src_size'] = src_size
        print(f"  Source Array Size: {src_size}")
        return memcpy_info
    return None

def query_count_declaration(tree, code):
    count_query_string = """
    (declaration
        declarator: (init_declarator
            declarator: (identifier) @count_var
            value: (number_literal) @count_value
        )
    )
    """
    query = C_LANGUAGE.query(count_query_string)
    captures = query.captures(tree.root_node)

    count_vars = {}
    for node, capture_name in captures:
        if capture_name == "count_var":
            var_name = code[node.start_byte:node.end_byte].strip()
        elif capture_name == "count_value":
            var_value = int(code[node.start_byte:node.end_byte].strip())
            count_vars[var_name] = var_value
            print(f"Count variable detected: {var_name} with value {var_value}")
    return count_vars
 
def query_malloc_allocation(tree, code):
    malloc_query_string = """
    (call_expression
        function: (identifier) @func_name
        arguments: (argument_list
            (binary_expression
                left: (identifier) @count_var
                operator: "*"
                right: (sizeof_expression
                    type: (type_descriptor
                        type: (primitive_type) @type
                    )
                )
            )
        )
    )
    """
    query = C_LANGUAGE.query(malloc_query_string)
    captures = query.captures(tree.root_node)

    malloc_calls = []
    for node, capture_name in captures:
        if capture_name == "count_var":
            count_var = code[node.start_byte:node.end_byte].strip()
        elif capture_name == "type":
            data_type = code[node.start_byte:node.end_byte].strip()
            malloc_calls.append((count_var, data_type))
            print(f"Malloc call detected with count variable {count_var} and type {data_type}")
    return malloc_calls

def query_if_condition(tree, code):
    if_query_string = """
    (if_statement
        condition: (parenthesized_expression
            (identifier) @condition_var
        )
    )
    """
    query = C_LANGUAGE.query(if_query_string)
    captures = query.captures(tree.root_node)

    if_conditions = []
    for node, capture_name in captures:
        if capture_name == "condition_var":
            condition_var = code[node.start_byte:node.end_byte].strip()
            if_conditions.append(condition_var)
            print(f"If condition detected: {condition_var}")
    return if_conditions


def query_return_statements(tree, code):
    return_query_string = """
    (return_statement
        (identifier) @return_var
    )
    """
    query = C_LANGUAGE.query(return_query_string)
    captures = query.captures(tree.root_node)

    return_vars = []
    for node, capture_name in captures:
        if capture_name == 'return_var':
            var_name = code[node.start_byte:node.end_byte].strip()
            return_vars.append(var_name)
    return return_vars

def query_malloc_calls_for_heap_allocation(tree, code):
    malloc_query_string = """
    (call_expression
        function: (identifier) @func_name
        arguments: (argument_list
            (binary_expression
                left: (number_literal) @alloc_size
                operator: "*"
                right: (sizeof_expression)
            )
        )
    )
    """
    query = C_LANGUAGE.query(malloc_query_string)
    captures = query.captures(tree.root_node)

    malloc_info = {}
    for node, capture_name in captures:
        text = code[node.start_byte:node.end_byte].strip()
        malloc_info[capture_name] = text

    if 'func_name' in malloc_info and malloc_info['func_name'] == 'malloc':
        return malloc_info
    return None

def query_write_operations_for_heap_overflow(tree, code):
    write_query_string = """
    (call_expression
        function: (identifier) @func_name
        arguments: (argument_list
            (identifier) @dest
            (string_literal) @src
        )
    )
    """
    query = C_LANGUAGE.query(write_query_string)
    captures = query.captures(tree.root_node)

    write_info = {}
    for node, capture_name in captures:
        text = code[node.start_byte:node.end_byte].strip()
        write_info[capture_name] = text

    if 'func_name' in write_info and write_info['func_name'] == 'strcpy':
        return write_info
    return None

def query_nested_loops(tree, code):
    nested_loop_query = """
    (for_statement
        initializer: (declaration) @init
        condition: (binary_expression
            left: (identifier) @loop_var
            right: (identifier) @loop_limit
        )
        update: (update_expression) @update
    )
    """

    query = C_LANGUAGE.query(nested_loop_query)
    captures = query.captures(tree.root_node)

    all_loops_info = []
    current_loop = {}
    for node, capture_name in captures:
        if capture_name == "init":
            current_loop['init'] = code[node.start_byte:node.end_byte].strip()
        elif capture_name == "loop_var":
            current_loop['loop_var'] = code[node.start_byte:node.end_byte].strip()
        elif capture_name == "loop_limit":
            current_loop['loop_limit'] = code[node.start_byte:node.end_byte].strip()
        elif capture_name == "update":
            current_loop['update'] = code[node.start_byte:node.end_byte].strip()
            all_loops_info.append(current_loop)
            current_loop = {}

    print("All Nested Loop analysis:")
    for loop_info in all_loops_info:
        print(f"  Initializer: {loop_info.get('init', '')}")
        print(f"  Loop Variable: {loop_info.get('loop_var', '')}")
        print(f"  Loop Limit: {loop_info.get('loop_limit', '')}")
        print(f"  Update: {loop_info.get('update', '')}")
    print()
    
    return all_loops_info

def query_multidimensional_arrays(tree, code):
    array_query = """
    (declaration
        declarator: (array_declarator) @array
    )
    """

    query = C_LANGUAGE.query(array_query)
    captures = query.captures(tree.root_node)

    arrays_info = {}
    for node, capture_name in captures:
        array_name = code[node.start_byte:node.end_byte].split('[')[0].strip()  # 提取数组名称
        sizes = [int(size.strip(']')) for size in code[node.start_byte:node.end_byte].split('[')[1:] if size.endswith(']')]
        
        if array_name not in arrays_info:
            arrays_info[array_name] = sizes

    print("Multi-dimensional Array declaration analysis:")
    for array_name, dimensions in arrays_info.items():
        print(f"  Array Name: {array_name}, Dimensions: {dimensions}")
    print()
    
    return arrays_info


def query_function_call_argument(tree, code, function_name, argument_index):
    """
    查询函数调用中特定参数的值。
    """
    query_string = """
    (call_expression
        function: (identifier) @func_name
        arguments: (argument_list) @args
    )
    """
    query = C_LANGUAGE.query(query_string)
    captures = query.captures(tree.root_node)

    for i in range(0, len(captures), 2):  # Step by 2 to match function and arguments
        func_name = code[captures[i][0].start_byte:captures[i][0].end_byte]
        if func_name == function_name:
            args_node = captures[i + 1][0]  # Capture `argument_list`
            arg_nodes = args_node.children  # Get all argument children
            if argument_index < len(arg_nodes):  # Check if index is within bounds
                arg_node = arg_nodes[argument_index]
                arg_value = code[arg_node.start_byte:arg_node.end_byte]
                print(f"Function '{function_name}' called with argument '{arg_value}' at position {argument_index}")
                return arg_value

    return None

def query_realloc_calls(tree, code):
    realloc_query_string = """
    (call_expression
        function: (identifier) @func_name
        arguments: (argument_list
            (identifier) @buffer_name
            (identifier) @new_size
        )
    )
    """
    query = C_LANGUAGE.query(realloc_query_string)
    captures = query.captures(tree.root_node)

    realloc_info = []
    for node, capture_name in captures:
        text = code[node.start_byte:node.end_byte].strip()
        if capture_name == 'func_name' and text == 'realloc':
            buffer_name = code[captures[1][0].start_byte:captures[1][0].end_byte].strip()
            new_size = code[captures[2][0].start_byte:captures[2][0].end_byte].strip()
            realloc_info.append({
                "func_name": text,
                "buffer_name": buffer_name,
                "new_size": new_size
            })
    return realloc_info

def query_stack_variables_and_return(tree, code):
    stack_var_query_string = """
    (function_definition
        body: (compound_statement
            (declaration
                declarator: (array_declarator declarator: (identifier) @stack_var)
            )*
            (return_statement (identifier) @return_var)
        )
    )
    """
    query = C_LANGUAGE.query(stack_var_query_string)
    captures = query.captures(tree.root_node)

    stack_vars = set()
    return_var = None

    for node, capture_name in captures:
        if capture_name == "stack_var":
            stack_var_name = code[node.start_byte:node.end_byte].strip()
            stack_vars.add(stack_var_name)
        elif capture_name == "return_var":
            return_var = code[node.start_byte:node.end_byte].strip()

    return stack_vars, return_var

def query_malloc_with_multiplication(tree, code):
    malloc_query_string = """
    (call_expression
        function: (identifier) @func_name
        arguments: (argument_list
            (binary_expression
                left: (binary_expression
                    left: (identifier) @count
                    operator: "*"
                    right: (identifier) @size
                )
                operator: "*"
                right: (sizeof_expression)
            )
        )
    )
    """
    query = C_LANGUAGE.query(malloc_query_string)
    captures = query.captures(tree.root_node)
    
    malloc_info = {}
    for node, capture_name in captures:
        if capture_name == "func_name" and code[node.start_byte:node.end_byte].strip() == "malloc":
            malloc_info['count'] = code[captures[1][0].start_byte:captures[1][0].end_byte].strip()
            malloc_info['size'] = code[captures[2][0].start_byte:captures[2][0].end_byte].strip()
            return malloc_info
    return None


def query_for_loop_out_of_bounds(tree, code):
    loop_query_string = """
    (for_statement
        initializer: (declaration) @init
        condition: (binary_expression
            left: (identifier) @loop_var
            right: (binary_expression
                left: (identifier) @size_var
                operator: "+"
                right: (number_literal) @offset
            )
        )
        update: (update_expression) @update
    )
    """
    query = C_LANGUAGE.query(loop_query_string)
    captures = query.captures(tree.root_node)

    loop_info = {}
    for node, capture_name in captures:
        if capture_name == 'loop_var':
            loop_info['loop_var'] = code[node.start_byte:node.end_byte].strip()
        elif capture_name == 'size_var':
            loop_info['size_var'] = code[node.start_byte:node.end_byte].strip()
        elif capture_name == 'offset':
            loop_info['offset'] = int(code[node.start_byte:node.end_byte].strip())
        elif capture_name == 'update':
            loop_info['update'] = code[node.start_byte:node.end_byte].strip()

    if 'size_var' in loop_info and 'offset' in loop_info:
        print(f"Warning: Potential out-of-bounds access detected! Loop accesses beyond allocated buffer size by offset {loop_info['offset']}.")
    else:
        print("No out-of-bounds access detected in for-loop.")

def query_memcpy_extra_length(tree, code):
    """
    Queries the AST to find the extra length value added to strlen(src) in memcpy(dest, src, strlen(src) + 5).
    """
    memcpy_extra_length_query = """
    (call_expression
        function: (identifier) @func_name
        arguments: (argument_list
            (identifier) @dest
            (identifier) @src
            (binary_expression
                left: (call_expression
                    function: (identifier) @strlen_func
                    arguments: (argument_list (identifier) @strlen_arg)
                )
                operator: "+"
                right: (number_literal) @extra_length
            )
        )
    )
    """
    
    query = C_LANGUAGE.query(memcpy_extra_length_query)
    captures = query.captures(tree.root_node)
    
    # Find the extra_length value if available
    for node, capture_name in captures:
        if capture_name == "extra_length":
            extra_length = int(code[node.start_byte:node.end_byte].strip())
            print(f"The value of 'extra_length' is: {extra_length}")
            return extra_length
    
    print("No extra length found in memcpy usage.")
    return None


def query_string_value(tree, code, target_var_name="src"):

    query_string = f"""
    (declaration
        declarator: (init_declarator
            declarator: (pointer_declarator
                declarator: (identifier) @var_name
            )
            value: (string_literal) @string_value
        )
    )
    """
    query = C_LANGUAGE.query(query_string)
    captures = query.captures(tree.root_node)

    for node, capture_name in captures:
        if capture_name == 'var_name' and code[node.start_byte:node.end_byte].strip() == target_var_name:
            value_node = captures[captures.index((node, capture_name)) + 1][0]  
            variable_value = code[value_node.start_byte:value_node.end_byte].strip().strip('"')
            print(f"The value of '{target_var_name}' is: {variable_value}")
            return variable_value

    print(f"Variable '{target_var_name}' not found or not initialized with a string.")
    return None


'''
---------------------------------------------------------
---------------------------------------------------------
Process part for detect the problems
---------------------------------------------------------
---------------------------------------------------------
'''

# Buffer Overflow and Out-of-Bounds Access Detection
def check_for_out_of_bounds1(for_parts, array_parts):
    
    if 'loop_limit' in for_parts and 'size' in array_parts:
        loop_limit = int(for_parts['loop_limit'])
        array_size = int(array_parts['size'])
        
        if loop_limit >= array_size:
            print("Warning: Potential out-of-bounds access detected! Loop limit exceeds buffer size.")
        else:
            print("No out-of-bounds access detected.")
    else:
        print("Insufficient data to perform out-of-bounds check.")

def detect_array_access(tree, code, array_info):
    query_string = """
    (subscript_expression
        argument: (identifier) @array_name
        index: (number_literal) @index
    )
    """
    query = C_LANGUAGE.query(query_string)
    captures = query.captures(tree.root_node)
    
    print("Array Access Analysis:")
    for node, capture_name in captures:
        if capture_name == 'array_name' and code[node.start_byte:node.end_byte].strip() == array_info.get('name'):
            array_name = code[node.start_byte:node.end_byte].strip()
        elif capture_name == 'index':
            access_index = int(code[node.start_byte:node.end_byte].strip())
            array_size = int(array_info.get('size'))
            
            if access_index < 0:
                print(f"  Warning: Buffer under-read detected! Access index {access_index} is below array start.")
            elif access_index >= array_size:
                print(f"  Warning: Out-of-bounds read detected! Access index {access_index} exceeds array size {array_size}.")
            else:
                print("  Access within bounds.")
    print()

def detect_overflow_in_multidimensional_arrays(arrays_info, loops_info, len_value):

    for array_name, dimensions in arrays_info.items():
        for loop_info in loops_info:
            loop_var = loop_info.get('loop_var')
            loop_limit = loop_info.get('loop_limit')

            if loop_var and loop_limit:

                loop_limit_value = int(len_value)
                print(f"  Warning: Loop limit '{loop_limit}' for '{loop_var}' in '{array_name}' is dynamic.")

                if loop_limit_value > dimensions[0]:
                    print(f"  Warning: Out-of-bounds access detected in '{array_name}' array! Loop limit {loop_limit_value} exceeds array size {dimensions[0]}.")

    print()

def detect_multi_out_of_bounds_access(tree, code):
    # Query array declarations to get array names and sizes
    array_info = query_array_declaration(tree, code)
    
    # Check parameter values in function calls
    calls = [
        query_function_call_argument(tree, code, "read_mixed_patterns", 1)
    ]
    
    for array in array_info:
        array_name = array["name"]
        array_size = array["size"]

        for call_index, index in enumerate(calls):
            if index is None:
                continue
            
            try:
                index = int(index)
            except ValueError:
                print(f"  Dynamic or non-numeric index detected for call at position {call_index}: '{index}'")
                continue

            if index < 0:
                print(f"  Warning: Buffer under-read detected! Index {index} is below 0 for array '{array_name}'.")
            elif index >= array_size:
                print(f"  Warning: Out-of-bounds read detected! Index {index} exceeds array size {array_size} for array '{array_name}'.")
            else:
                print(f"  Access within bounds for index {index} on array '{array_name}'.")

def detect_buffer_overflow_and_overread(tree, code):

    target_functions = ["process_strings", "process_input"]
    buffer_info = query_array_declaration(tree, code)  
    buffer_size = None
    for array in buffer_info:
        if array.get("name") == "localBuffer" or "buffer":
            buffer_size = array.get("size")
            print(f"Detected localBuffer size: {buffer_size}")
            break

    length_value = None
    for function_name in target_functions:
        length_value = query_function_call_argument(tree, code, function_name, 3)
        if length_value is not None: 
            length_value = int(length_value)
            print(f"Detected length parameter value: {length_value} in function '{function_name}'")
            break

    if buffer_size is not None and length_value is not None and length_value > buffer_size:
        print(f"Warning: Potential buffer overflow! The length ({length_value}) exceeds localBuffer size ({buffer_size}).")
    
    if length_value is not None and length_value < 0:
        print("Warning Negative length causes under-read")

    data_info = query_array_declaration(tree, code)  
    data_size = None
    for array in data_info:
        if array.get("name") == "data":
            data_size = array.get("size")
            #print(f"Detected data size: {data_size}")
            break

    if data_size is not None and length_value > data_size:
        print(f"Warning: Potential buffer over-read! The length ({length_value}) exceeds data size ({data_size}).")

def detect_dynamic_overflow_in_loop(tree, code):
    malloc_size = query_malloc_calls_forvalue(tree, code)  # Get initial malloc size
    realloc_info = query_realloc_calls(tree, code)  # Get realloc call information
    final_size_arg = query_function_call_argument(tree, code, "dynamic_write", 3)

    if not malloc_size and not realloc_info and not final_size_arg:
        #print("Skipping dynamic overflow detection as no relevant allocation data is found.")
        return  # If no data is available, skip detection

    if final_size_arg and malloc_size[0] is not None:
        try:
            final_size = int(final_size_arg)  # Try to convert string to integer
        except ValueError:
            #print(f"Unable to interpret final size from argument '{final_size_arg}'. Skipping detection.")
            return  # Unable to convert, skip detection
    else:
        if realloc_info:
            try:
                final_size = int(realloc_info[0]['new_size'])
            except ValueError:
                #print(f"Unable to interpret realloc size '{realloc_info[0]['new_size']}'. Skipping detection.")
                return  # Unable to convert, skip detection
        else:
            final_size = int(malloc_size[0])

    loop_limit = final_size
    if loop_limit is not None:
        if loop_limit > int(malloc_size[0]):
            print("Warning: Potential buffer overflow detected! Loop limit exceeds allocated buffer size.")
        else:
            print("No overflow detected in loop access.")

def detect_memcpy_overflow_with_src(tree, code):
    """
    Detects potential buffer overflow in memcpy based on the `src` string literal.
    """
    # Step 1: Query the value of `src`
    src_value = query_string_value(tree, code)
    if src_value is None:
        print("No value found for `src`; skipping memcpy overflow detection.")
        return

    # Step 2: Query extra_length in memcpy
    extra_length = query_memcpy_extra_length(tree, code)
    if extra_length is None:
        print("No extra length detected in memcpy; skipping overflow detection.")
        return

    # Parameters and detection logic
    dest_size = 10  
    src_length = len(src_value)
    total_copy_length = src_length + extra_length

    if total_copy_length > dest_size:
        print(f"Warning: Potential buffer overflow in memcpy! Trying to copy {total_copy_length} bytes into {dest_size}-byte destination buffer.")
    else:
        print("No buffer overflow detected in memcpy usage.")

def check_memcpy_overflow(memcpy_info, arrays_info):
    dest_name = memcpy_info.get('dest')
    dest_size = next((array['size'] for array in arrays_info if array['name'] == dest_name), None)
    src_size = memcpy_info.get('src_size')

    print("\nMemcpy Overflow Detection:")
    if dest_size is None or src_size is None:
        print("  Unable to perform memcpy overflow check due to missing size information.")
        return

    if dest_size < src_size:
        print(f"  Warning: Potential overflow in memcpy detected! Trying to copy {src_size} bytes into {dest_size}-byte destination buffer.")
    else:
        print("  No overflow detected in memcpy usage.")

def detect_realloc_overflow(tree, code):
    malloc_sizes = query_malloc_calls_forvalue(tree, code)
    if malloc_sizes:
        malloc_size = int(malloc_sizes[0])  
        print(f"Detected malloc size: {malloc_size}")
    else:
        malloc_size = None  

    realloc_size = query_function_call_argument(tree, code, "reallocate_buffer", 1)
    print(realloc_size)
    if realloc_size is not None:
        realloc_size = int(realloc_size)  
        print(f"Detected realloc new size: {realloc_size}")
    else:
        realloc_size = None  

    loop_info = query_nested_loops(tree, code)
    if loop_info:  
        loop_limit = loop_info[0].get('loop_limit', None) 
        if loop_limit is not None:
            #print(f"Loop limit detected: {loop_limit}")
            
            if loop_limit == "newSize": 
                loop_limit = realloc_size  
                print(f"Dynamic loop limit detected and set to: {loop_limit}")
            else:
                try:
                    loop_limit = int(loop_limit)  
                except ValueError:
                    #print(f"Unresolved dynamic loop limit: {loop_limit}")
                    loop_limit = None
    else:
        loop_limit = None  

    if malloc_size is not None and realloc_size is not None and loop_limit is not None:
        if loop_limit > malloc_size:
            print("Warning: Potential overflow detected! Loop limit exceeds initial malloc size.")
        
        if loop_limit > realloc_size:
            print("Warning: Potential overflow detected! Loop limit exceeds reallocated buffer size.")

# Memory Management Vulnerabilities Detection
def detect_heap_overflow(malloc_info, write_info):
    #print("Heap Overflow Detection:")
    if not malloc_info or not write_info:
        #print("  Insufficient data for heap overflow detection.")
        return
    
    alloc_size = int(malloc_info.get('alloc_size', 0))
    src_length = len(write_info.get('src', ''))

    if alloc_size < src_length:
        print(f"  Warning: Potential heap buffer overflow detected! Attempt to write {src_length} bytes into {alloc_size}-byte buffer.")
    else:
        print("  No overflow detected in heap allocation.")

def detect_double_free(malloc_vars, free_vars):
    #print("Double Free Analysis:")
    allocated_vars = set(malloc_vars)
    freed_vars = set()

    for var in free_vars:
        if var not in allocated_vars:
            print(f"  Warning: Freeing unallocated variable '{var}'.")
        elif var in freed_vars:
            print(f"  Warning: Double free detected for variable '{var}'.")
        else:
            freed_vars.add(var)

def detect_use_after_free(malloc_vars, free_vars, dereference_vars):
    #print("Use After Free Analysis:")
    freed = set(free_vars)
    for var in dereference_vars:
        if var in freed:
            print(f"  Warning: Potential use after free detected for variable '{var}'.")

def detect_use_after_free_in_conditional(tree, code):
    # Get all allocated variables via malloc
    malloc_vars = query_malloc_calls_for_valuename(tree, code)
    
    # Get all variables freed via free
    free_vars = query_free_calls(tree, code)
    
    # Query pointer accesses within conditional statements
    dereference_query_string = """
    (if_statement
        condition: (parenthesized_expression (_) @condition)
        consequence: (compound_statement
            (expression_statement
                (assignment_expression
                    left: (subscript_expression
                        argument: (identifier) @var_name
                    )
                )
            )
        )
    )
    """
    query = C_LANGUAGE.query(dereference_query_string)
    captures = query.captures(tree.root_node)

    # Check if any freed variable is used within conditional statements
    freed_vars = set(free_vars)
    #print("Conditional Use-After-Free Analysis:")
    for node, capture_name in captures:
        if capture_name == 'var_name':
            var_name = code[node.start_byte:node.end_byte].strip()
            if var_name in freed_vars:
                print(f"Warning: Use-after-free detected for variable '{var_name}' in a conditional block.")
            else:
                print(f"No use-after-free detected for variable '{var_name}' in this conditional block.")

def detect_return_of_stack_variable(stack_vars, return_var):
    # Check if the returned variable is a local stack variable
    if return_var and return_var in stack_vars:
        print(f"Warning: Returning local stack variable '{return_var}', which may lead to invalid memory access.")
    else:
        print("No issues detected with return value.")

def detect_invalid_return(arrays_info, return_vars):
    #print("Invalid Pointer Return Analysis:")
    for array in arrays_info:
        if array["name"] in return_vars:
            print(f"  Warning: Returning local stack-allocated array '{array['name']}' from function. This can lead to undefined behavior.")

# Integer Overflow Detection
def check_for_integer_overflow(count_vars, malloc_calls):
    overflow_detected = False
    MAX_32BIT_SIZE = 2**31 - 1 
    for count_var, data_type in malloc_calls:

        count_value = count_vars.get(count_var)

        sizeof_value = 4 if data_type == 'int' else None

        if count_value is not None and sizeof_value is not None:
            total_size = count_value * sizeof_value
            #print(f"Calculated total allocation size: {total_size}")
            if total_size > MAX_32BIT_SIZE:  
                print(f"Warning: Potential integer overflow detected for count variable '{count_var}', allocation size {total_size} exceeds 32-bit limit.")
                overflow_detected = True
            else:
                print(f"No overflow detected for count variable '{count_var}' with total allocation size {total_size}.")
        else:
            print("Insufficient data to perform overflow check.")

    return overflow_detected

def detect_integer_overflow_and_out_of_bounds(tree, code):
    malloc_info = query_malloc_with_multiplication(tree, code)
    
    if malloc_info:


        count_value_str = query_function_call_argument(tree, code, "calculate_overflow", 1)
        size_value_str = query_function_call_argument(tree, code, "calculate_overflow", 3)

        count_value = int(count_value_str)
        size_value = int(size_value_str) 
        int_max = 2**31 - 1  # Max value for 32-bit integer

        # Check for multiplication overflow
        if count_value * size_value > int_max:
            print(f"Warning: Potential integer overflow detected! count * size exceeds 32-bit integer limit.")

        # Check for out-of-bounds write when allocation fails
        allocation_size = count_value * size_value * 4  # Assuming sizeof(int) is 4
        if allocation_size > int_max or allocation_size <= 0:
            print("Warning: Allocation failed due to overflow, leading to potential out-of-bounds write.")
        else:
            print("No issues detected in allocation size.")
    else:
        print("No malloc with multiplication detected.")

# Main Detection Flow
def main_detection_flow(tree, code):
    count_vars = query_count_declaration(tree, code)
    malloc_calls = query_malloc_allocation(tree, code)
    overflow_detected = check_for_integer_overflow(count_vars, malloc_calls)
    if overflow_detected:
        print("Potential integer overflow detected that could lead to buffer overflow.")
    else:
        print("No overflow issues detected in malloc usage.")
    
    malloc_info = query_malloc_calls_for_heap_allocation(tree, code)
    write_info = query_write_operations_for_heap_overflow(tree, code)
    detect_heap_overflow(malloc_info, write_info)
    
    # Detect out-of-bounds access
    for_parts = query_loop(tree, code)
    array_parts = query_array_declaration(tree, code)
    if array_parts and for_parts:
        check_for_out_of_bounds1(for_parts, array_parts[0])
    

    # Detect out-of-bounds array access
    if array_parts:
        detect_array_access(tree, code, array_parts[0])
    
    # Detect double free
    malloc_vars = query_malloc_calls_for_valuename(tree, code)
    print("Malloc Variables Detected:", malloc_vars)
    free_vars = query_free_calls(tree, code)

    # Pass realloc_vars to detect_double_free
    detect_double_free(malloc_vars, free_vars)
    
    # Detect use after free
    dereference_vars = query_dereference(tree, code)
    detect_use_after_free(malloc_vars, free_vars, dereference_vars)

    # Detect invalid return of local arrays
    return_vars = query_return_statements(tree, code)
    detect_invalid_return(array_parts, return_vars)

    arrays_info = query_array_declaration(tree, code)
    memcpy_info = query_memcpy_usage(tree, code)
    if memcpy_info:
        check_memcpy_overflow(memcpy_info, arrays_info)

    len_value = query_function_call_argument(tree, code, "nested_overflow",3)
    multi_loops_info = query_nested_loops(tree, code)
    multi_arrays_info = query_multidimensional_arrays(tree, code)
    detect_overflow_in_multidimensional_arrays(multi_arrays_info, multi_loops_info, len_value)

    detect_realloc_overflow(tree, code)
    detect_multi_out_of_bounds_access(tree, code)
    # Call detection functions
    detect_buffer_overflow_and_overread(tree, code)

    detect_use_after_free_in_conditional(tree, code)

    stack_vars, return_var = query_stack_variables_and_return(tree, code)

    detect_return_of_stack_variable(stack_vars, return_var)

    detect_integer_overflow_and_out_of_bounds(tree, code)

    detect_dynamic_overflow_in_loop(tree, code)

    query_for_loop_out_of_bounds(tree, code)

    detect_memcpy_overflow_with_src(tree, code)

# Run main detection function
main_detection_flow(tree, code)
