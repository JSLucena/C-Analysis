void simpleDeclarations() {
    int x;
    float y;
    char z = 'A';
    int arr[5];
}

void declarationsWithAssignment() {
    int a = 10;
    float b = 5.5;
    int arr[3] = {1, 2, 3};
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
    if (x > 0) {
        x -= 1;
    } else {
        x = 0;
    }
}

void binaryExpressions() {
    int m = 10, n = 20;
    int result = m + n * 2;
}

void complexExpressions(int x, int y, int z, int w) {
    int a = (x + y) * (z - 3) / (w + 1);
}
