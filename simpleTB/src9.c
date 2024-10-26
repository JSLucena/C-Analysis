#include <stdlib.h>
#include <stdio.h>

//CWE-680: Integer Overflow to Buffer Overflow
//CWE-122: Heap-based Buffer Overflow


void intOverflow() {
    int count = 1073741824; // 2^30
    int *buffer = malloc(count * sizeof(int)); // Integer overflow here
    if (buffer) {
        buffer[0] = 1; // May cause a crash or buffer overflow
        free(buffer);
    }
}

int main() {
    intOverflow();
    return 0;
}