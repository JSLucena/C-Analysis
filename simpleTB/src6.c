#include <stdlib.h>

//CWE-415: Double Free

void doubleFree() {
    int *ptr = (int *)malloc(sizeof(int));
    free(ptr);
    free(ptr); // Double free
}

int main() {
    doubleFree();
    return 0;
}