#include <stdio.h>
#include <string.h>

void process_input(char *data, int length) {
    char buffer[10] = {0};
    if (length > 0) {
        // CWE-127: Buffer Under-read if negative length is passed
        // CWE-126: Buffer Over-read if length exceeds data size
        strncpy(buffer, data - length, length); // Read before data start
        printf("Processed: %s\n", buffer);
    }
}

int main() {
    char data[] = "Example";
    process_input(data, -2); // Negative length causes under-read
    return 0;
}
