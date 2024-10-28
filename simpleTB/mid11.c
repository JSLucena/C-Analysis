#include <stdio.h>
#include <stdlib.h>

void complex_access(char *buffer, int size) {
    // CWE-788: Access of Memory Location After End of Buffer
    for (int i = 0; i < size + 5; i++) {
        printf("%c", buffer[i]); // Out-of-bounds access when i >= size
    }
}

int main() {
    char *buffer = (char *)malloc(10);
    snprintf(buffer, 10, "Test1234");
    complex_access(buffer, 10); // Out-of-bounds access
    free(buffer);
    return 0;
}
