#include <stdio.h>
#include <stdlib.h>

void reallocate_buffer(int newSize) {
    char *buffer = (char *)malloc(10);
    // CWE-122: Heap-based Buffer Overflow
    // CWE-680: Integer Overflow to Buffer Overflow if newSize is large
    buffer = realloc(buffer, newSize); // Potentially reallocates to smaller size
    for (int i = 0; i < newSize; i++) {
        buffer[i] = 'A'; // Out-of-bounds write if reallocated buffer is smaller
    }
    free(buffer);
}

int main() {
    reallocate_buffer(20); // Unsafe if newSize is uncontrolled
    return 0;
}
