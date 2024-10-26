#include <stdlib.h>
#include <stdio.h>

//CWE-416: Use After Free


void useAfterFree() {
    int *ptr = (int *)malloc(sizeof(int));
    free(ptr);
    *ptr = 10; // Access after free
    printf("%d\n", *ptr);
}

int main() {
    useAfterFree();
    return 0;
}