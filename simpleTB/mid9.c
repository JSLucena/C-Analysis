#include <stdlib.h>
#include <stdio.h>

void calculate_overflow(int count, int size) {
    int *array = NULL;
    if (count > 0 && size > 0) {
        // CWE-680: Integer Overflow to Buffer Overflow if count * size overflows
        array = (int *)malloc(count * size * sizeof(int));
        if (array) {
            array[0] = 1; // Out-of-bounds write if allocation fails
        }
        free(array);
    }
}

int main() {
    calculate_overflow(100000, 100000); // Potential overflow
    return 0;
}
