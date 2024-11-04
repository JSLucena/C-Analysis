#include <stdio.h>

char* get_string(int flag) {
    static char global_buffer[10] = "Global";
    char stack_buffer[10] = "Stack";
    // CWE-466: Return of Pointer Value Outside of Expected Range
    if(flag)
        return stack_buffer;
    else
        return global_buffer;
}

int main() {
    char *str = get_string(1);
    printf("String: %s\n", str); // Potentially invalid memory access
    return 0;
}
