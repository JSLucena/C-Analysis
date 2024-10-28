#include <stdlib.h>
#include <string.h>

//CWE-122: Heap-based Buffer Overflow
//CWE-787: Out-of-Bounds Write


void heapOverflow() {
    char *buffer = (char *)malloc(10 * sizeof(char));
    strcpy(buffer, "This is too long for buffer"); // Overflows buffer
    free(buffer);
}

int main() {
    heapOverflow();
    return 0;
}
