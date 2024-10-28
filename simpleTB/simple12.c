#include <stdio.h>
#include <string.h>

//CWE-805: Buffer Access with Incorrect Length Value
//CWE-122: Heap-based Buffer Overflow (if buffer is dynamically allocated)
//CWE-787: Out-of-Bounds Write (if write occurs beyond allocated space)
//CWE-126: Buffer Over-read (if it causes over-read)


void incorrectLength() {
    char src[] = "This is a long string";
    char dest[10];
    memcpy(dest, src, sizeof(src)); // Copies too many bytes into dest
}

int main() {
    incorrectLength();
    return 0;
}