#include <stdio.h>


//CWE-127: Buffer Under-read
//CWE-125: Out-of-Bounds Read

void bufferUnderread() {
    char buffer[5] = "Hello";
    printf("%c\n", buffer[-1]); // Under-reads the buffer
}

int main() {
    bufferUnderread();
    return 0;
}