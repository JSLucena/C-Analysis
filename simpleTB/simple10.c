#include <stdio.h>

//CWE-787: Out-of-Bounds Write
//CWE-121: Stack-based Buffer Overflow (if it's on stack)


void outOfBoundsWrite() {
    int buffer[5];
    for (int i = 0; i <= 5; i++) {
        buffer[i] = i; // Writes out of bounds on i = 5
    }
}

int main() {
    outOfBoundsWrite();
    return 0;
}