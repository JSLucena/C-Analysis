#include <stdio.h>

//CWE-121: Stack-based Buffer Overflow
//CWE-787: Out-of-Bounds Write


void stackOverflow() {
    char buffer[10];
    for (int i = 0; i < 15; i++) {
        buffer[i] = 'A'; // Overflows the buffer when i >= 10
    }
}

int main() {
    stackOverflow();
    return 0;
}