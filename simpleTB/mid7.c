#include <stdio.h>
#include <stdlib.h>

void conditional_use_after_free(int flag) {
    int *data = (int *)malloc(5 * sizeof(int));
    free(data);
    if (flag) {
        // CWE-416: Use After Free if flag == 1
        // CWE-787: Out-of-Bounds Write if buffer is freed
        data[0] = 10; // Writes to memory after free
    }
}

int main() {
    conditional_use_after_free(1); // Triggers use-after-free
    return 0;
}
