#include <stdio.h>
#include <string.h>

void copy_with_incorrect_length(char *src) {
    char dest[10];
    // CWE-805: Buffer Access with Incorrect Length Value
    // CWE-126: Buffer Over-read if src is smaller than length
    memcpy(dest, src, strlen(src) + 5); // Reads past end of src
    dest[9] = '\0';
    printf("Destination: %s\n", dest);
}

int main() {
    char *src = "1234567890";
    copy_with_incorrect_length(src); // Incorrect length causes overflow
    return 0;
}
