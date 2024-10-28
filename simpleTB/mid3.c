#include <stdio.h>

void read_mixed_patterns(int index) {
    int buffer[10] = {0};
    // CWE-125: Out-of-Bounds Read
    // CWE-127: Buffer Under-read if index < 0
    if (index >= 0 && index < 10) {
        printf("%d ", buffer[index]);
    } else {
        printf("Out-of-bounds read: %d\n", buffer[index]); // Out-of-bounds read
    }
}

int main() {
    read_mixed_patterns(15); // Invalid index access
    read_mixed_patterns(-3); // Under-read
    return 0;
}
