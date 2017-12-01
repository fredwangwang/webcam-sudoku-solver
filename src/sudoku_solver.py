'''Using a constraint SAT solver to solve sudoku'''

from z3 import Int, And, Distinct, If, Solver, sat, print_matrix

# 9x9 matrix of integer variables
X = [[Int("x_%s_%s" % (i + 1, j + 1)) for j in range(9)]
     for i in range(9)]

# each cell contains a value in {1, ..., 9}
cells_c = [And(1 <= X[i][j], X[i][j] <= 9)
           for i in range(9) for j in range(9)]

# each row contains a digit at most once
rows_c = [Distinct(X[i]) for i in range(9)]

# each column contains a digit at most once
cols_c = [Distinct([X[i][j] for i in range(9)])
          for j in range(9)]

# each 3x3 square contains a digit at most once
sq_c = [Distinct([X[3 * i0 + i][3 * j0 + j]
                  for i in range(3) for j in range(3)])
        for i0 in range(3) for j0 in range(3)]

sudoku_c = cells_c + rows_c + cols_c + sq_c

s = Solver()
s.add(sudoku_c)


def solve(grid):
    '''Solve the sudoku using Z3 solver. If a solution find, return the solution, otherwise return None

    Input: 2d grid of the raw sudoku. empty cell written as 0

    solve(grid) -> solution
    '''
    instance_c = [If(grid[i][j] == 0,
                     True,
                     X[i][j] == grid[i][j])
                  for i in range(9) for j in range(9)]
    s.push()
    s.add(instance_c)
    if s.check() == sat:
        m = s.model()
        r = [[m.evaluate(X[i][j]) for j in range(9)]
             for i in range(9)]
        result = r
    else:
        result = None
    s.pop()

    return result


__instance = ((0, 7, 0, 8, 6, 4, 0, 3, 0),
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
    solution = solve(__instance)
    if not solution:
        print 'no solution'
    else:
        print_matrix(solution)
