#include <stdio.h>

void nested_overflow(char *input, int len) {
    char buffer[8][8];
    // CWE-121: Stack-based Buffer Overflow
    // CWE-787: Out-of-Bounds Write if len exceeds buffer size
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < 8; j++) {
            buffer[i][j] = input[j]; // Writes beyond buffer if i >= 8
        }
    }
}

int main() {
    char *input = "This string is too long";
    nested_overflow(input, 16); // Causes buffer overflow
    return 0;
}
