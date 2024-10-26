#include <stdio.h>

//CWE-466: Return of Pointer Value Outside of Expected Range

char *badPointer() {
    char buffer[10] = "Hello";
    return buffer; // Returning stack-allocated buffer
}

int main() {
    char *str = badPointer();
    printf("%s\n", str); // Undefined behavior
    return 0;
}