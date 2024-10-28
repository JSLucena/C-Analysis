#include <stdio.h>
#include <stdlib.h>

void allocate_memory(int flag) {
    char *buffer = (char *)malloc(20);
    if (flag) {
        free(buffer); // CWE-415: Double Free
    }
    // CWE-415: Double Free due to repeated freeing
    free(buffer); // Free called twice if flag == 1
}

int main() {
    allocate_memory(1);
    return 0;
}
