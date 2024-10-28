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
