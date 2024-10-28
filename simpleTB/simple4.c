#include <stdio.h>
#include <string.h>

//CWE-126: Buffer Over-read
//CWE-125: Out-of-Bounds Read


void bufferOverread() {
    char buffer[10] = "Hello";
    for (int i = 0; i < 15; i++) {
        printf("%c", buffer[i]); // Over-reads the buffer
    }
    printf("\n");
}

int main() {
    bufferOverread();
    return 0;
}
