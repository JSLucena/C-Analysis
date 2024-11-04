from z3 import *

def AtMostOne(probs):
  return Or(
    And(*[Not(q) for q in probs]),
    *[
        And(p, *[Not(q) for iq, q in enumerate(probs) if iq != ip])
        for ip, p in enumerate(probs)
    ],
    )

def solve_n_queens(n):
    # Create boolean variables for each square
    queens = [[Bool(f"Q_{i}_{j}") for j in range(n)] for i in range(n)]

    # Create a solver
    s = Solver()

    # Constraint 1: Exactly one queen per row
    for i in range(n):
        s.add(PbEq([(queens[i][j], 1) for j in range(n)], 1))

    # Constraint 2: Exactly one queen per column
    for j in range(n):
        s.add(PbEq([(queens[i][j], 1) for i in range(n)], 1))

    # Constraint 3: At most one queen per diagonal
    # Diagonals going up-right
    for i in range(n - 1):
        for j in range(n - i - 1):
            s.add(AtMostOne([queens[i+k][j+k] for k in range(n - i - j)]))

    # Diagonals going up-left
    for i in range(n - 1):
        for j in range(i + 1, n):
            s.add(AtMostOne([queens[i-k][j-k] for k in range(i + 1)]))

    # Check if the solver can find a solution
    if s.check() == sat:
        m = s.model()
        for i in range(n):
            for j in range(n):
                if m.evaluate(queens[i][j]):
                    print("Q", end="")
                else:
                    print(".", end="")
            print()
    else:
        print("No solution found")

# Solve the 4-Queens problem
solve_n_queens(24)