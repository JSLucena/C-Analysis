void simpleDeclarations() {
    int x;
    float y;
    char z = 'A';
    char s[10] = "Hello";
    int arr[5];
}

void declarationsWithAssignment() {
    int a = 10;
    float b = 5.5;
    int arr[3] = {1, 2, 3};
    int arr[3][3][3] = {1, 2, 3, 
    4, 5, 6, 
    7, 8, 9};
}

void basicForLoop() {
    int x = 0;
    for (int i = 0; i < 10; i++) {
        x += i;
    }
}

void nestedForLoop() {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            int k = i * j;
        }
    }
}



void conditionalBlock(int x) {
    if (x + 1 > 0) {
        x -= 1;
    } else {
        x = 0;
    }
}

void binaryExpressions() {
    int a,b;
    int m = 10, n = 20;
    int result = m + n * 2;
}

void complexExpressions(int x, int y, int z, int w) {
    int a = (x + y) * (z - 3) / (w + 1);
}

void forLoopWithBreakContinue() {
    for (int i = 0; i < 10; i++) {
        if (i == 5) {
            continue;
        }
        if (i == 8) {
            break;
        }
    }
}

void whileLoops() {
    int i = 0;
    while (i < 5) {
        i++;
    }

    int j = 5;
    do {
        j--;
    } while (j > 0);
}   

void complexLoop() {
    int i, j, sum = 0, product = 1;

    for (i = 1; i <= 5; i++) {
        sum += i;
        product *= i;

        // Conditional check and assignment
        if (i % 2 == 0) {
            product *= 2;
        } else {
            product *= 3;
        }

        // Array indexing and modification
        int arr[5] = {10, 20, 30, 40, 50};
        arr[i-1] += i;
    }

    printf("Sum: %d\nProduct: %d\n", sum, product);
}

void test3DArray() {
    int array3D[2][3][4];           // 3D array of integers
    array3D[1][2][3] = 42;          // Assign 42 to specific element
    array3D[0][0][0] = array3D[1][2][3];  // Assign using another array element
}

void testPointerDereference() {
    int x = 5;
    int *ptr = &x;           // Pointer to x
    int **ptr2 = &ptr;       // Pointer to pointer
    *ptr = 10;               // Dereference ptr and set x to 10
    **ptr2 = 15;             // Dereference ptr2 to set x to 15
}

void testPointerArithmetic() {
    int arr[5] = {1, 2, 3, 4, 5};
    int *ptr = arr;
    int *ptr_offset = ptr + 2;    // Pointer arithmetic
    int value = *(ptr + 3);       // Dereference with offset
}

void testPointerToArray() {
    int matrix[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int (*ptr_matrix)[3] = matrix;     // Pointer to an array of 3 ints
    int value = *(*(ptr_matrix + 1) + 2);  // Access matrix[1][2] through pointer
}

void testPointerDifference() {
    int numbers[4] = {10, 20, 30, 40};
    int *p1 = numbers;
    int *p2 = numbers + 2;
    int diff = p2 - p1;           // Difference between pointers
}

void testPointerToArrayElement() {
    int grid[2][2] = {{1, 2}, {3, 4}};
    int *row = grid[1];            // Pointer to the second row
    int item = *(row + 1);         // Access grid[1][1] through pointer offset
}
