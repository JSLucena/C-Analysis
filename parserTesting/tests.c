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