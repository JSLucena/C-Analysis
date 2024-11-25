import re

def extract_function(source_code: str, function_name: str) -> str:
    """
    Extract a function from source code, supporting various function signatures, including nested structures.
    
    Args:
        source_code (str): The complete source code containing the target function
        function_name (str): Name of the function to extract
    
    Returns:
        str: The extracted function code, or None if not found
    """
    # Pattern to find the function signature and start of the function body
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
        # Match opening brace of the function body
        r'\{'
    )

    # Compile and search for the function signature
    compiled_pattern = re.compile(pattern, re.VERBOSE | re.MULTILINE | re.DOTALL)
    match = compiled_pattern.search(source_code)

    if not match:
        return None

    # Start extracting from the opening brace
    start_index = match.start()
    open_braces = 0
    in_function = False

    # Traverse the source code from the match to find the function body
    for i in range(start_index, len(source_code)):
        if source_code[i] == '{':
            if not in_function:
                in_function = True
            open_braces += 1
        elif source_code[i] == '}':
            open_braces -= 1
            if open_braces == 0 and in_function:
                # Extract the function from start_index to current position (inclusive)
                function_code = source_code[start_index:i + 1]
                
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

                # Return the cleaned-up function code
                return '\n'.join(lines)

    # If we reach here, the function was not properly closed
    return None

# Example usage
source_code = """
#include <stdlib.h>
#include <stdio.h>

void dynamic_write(char *data, int size) {
    char *buffer = (char *)malloc(10);
    if (size > 10) {
        buffer = realloc(buffer, size); // Dynamic resizing
    }
    // CWE-787: Out-of-Bounds Write if buffer < size
    for (int i = 0; i < size; i++) {
        buffer[i] = data[i]; // Writes beyond buffer if not resized properly
    }
    free(buffer);
}

int main() {
    char data[] = "This is too large for the buffer";
    dynamic_write(data, 30); // Unsafe write
    return 0;
}

"""

function_name = "dynamic_write"
extracted_function = extract_function(source_code, function_name)
print(extracted_function)
