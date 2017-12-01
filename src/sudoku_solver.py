'''Using a constraint SAT solver to solve sudoku'''

from z3 import Int, And, Distinct, If, Solver, sat, print_matrix
import time

# 9x9 matrix of integer variables
X = [[Int("x_%s_%s" % (i + 1, j + 1)) for j in range(9)]
     for i in range(9)]

# each cell contains a value in {1, ..., 9}
CELLS_C = [And(X[i][j] >= 1, X[i][j] <= 9)
           for i in range(9) for j in range(9)]

# each row contains a digit at most once
ROWS_C = [Distinct(X[i]) for i in range(9)]

# each column contains a digit at most once
COLS_C = [Distinct([X[i][j] for i in range(9)])
          for j in range(9)]

# each 3x3 square contains a digit at most once
SQUARE_C = [Distinct([X[3 * i0 + i][3 * j0 + j]
                      for i in range(3) for j in range(3)])
            for i0 in range(3) for j0 in range(3)]

SUDOKU_C = CELLS_C + ROWS_C + COLS_C + SQUARE_C

SOLVER = Solver()
SOLVER.add(SUDOKU_C)


def solve_sudoku(grid):
    '''Solve the sudoku using Z3 solver. If a solution find, return the solution, otherwise return None

    Input: 2d grid of the raw sudoku. empty cell written as 0

    solve(grid) -> solution
    '''
    instance_c = [If(grid[i][j] == 0, True, X[i][j] == grid[i][j])
                  for i in range(9) for j in range(9)]

    SOLVER.push()
    SOLVER.add(instance_c)
    if SOLVER.check() == sat:
        m = SOLVER.model()
        result = [[m.evaluate(X[i][j]) for j in range(9)] for i in range(9)]
    else:
        result = None
    SOLVER.pop()

    return result


SAMPLE_INSTANCE = ((0, 7, 0, 8, 6, 4, 0, 3, 0),
                   (0, 2, 0, 0, 0, 0, 0, 9, 0),
                   (0, 0, 1, 0, 0, 0, 5, 0, 0),
                   (0, 0, 0, 7, 0, 3, 0, 0, 0),
                   (0, 0, 0, 0, 4, 0, 0, 0, 0),
                   (0, 0, 0, 6, 1, 9, 0, 0, 0),
                   (0, 1, 4, 0, 3, 0, 7, 0, 0),
                   (2, 0, 0, 0, 0, 0, 0, 0, 6),
                   (8, 0, 3, 2, 0, 5, 0, 0, 1))

# debug
if __name__ == '__main__':
    start = time.time()
    for i in range(20):
        solution = solve_sudoku(SAMPLE_INSTANCE)
        if not solution:
            print 'no solution'
        else:
            print 'sat'
    print_matrix(solution)
    print time.time() - start
