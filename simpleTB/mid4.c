#include <stdio.h>
#include <string.h>

void process_strings(char *data, int length) {
    char localBuffer[10];
    // CWE-126: Buffer Over-read if length exceeds data size
    // CWE-121: Stack-based Buffer Overflow if data is larger than localBuffer
    strncpy(localBuffer, data, length); 
    localBuffer[9] = '\0';
    printf("Data: %s\n", localBuffer);
}

int main() {
    char data[] = "Short";
    process_strings(data, 15); // Over-read occurs with length > data size
    return 0;
}
