#include <stdio.h>

//CWE-125: Out-of-Bounds Read
//CWE-126: Buffer Over-read (if the read accesses bytes beyond the buffer bounds)

void outOfBoundsRead() {
    int buffer[5] = {1, 2, 3, 4, 5};
    printf("%d\n", buffer[10]); // Reads out of bounds
}

int main() {
    outOfBoundsRead();
    return 0;
}
