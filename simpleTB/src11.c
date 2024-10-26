#include <stdio.h>

//CWE-788: Access of Memory Location After End of Buffer
//CWE-125: Out-of-Bounds Read


void accessAfterEnd() {
    int buffer[5] = {1, 2, 3, 4, 5};
    printf("%d\n", buffer[5]); // Accesses out of bounds after buffer
}

int main() {
    accessAfterEnd();
    return 0;
}