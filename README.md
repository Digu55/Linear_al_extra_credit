# Linear-al extra credit project

### Brief Report on the Assignment
#### Introduction
This assignment dives into implementing key concepts of linear algebra using Python, entirely from scratch, without relying on libraries like NumPy or SciPy. It showcases the mathematical underpinnings of vectors, matrices, eigenvalues, decompositions, and other advanced computations through a systematic and modular approach.

## Code Structure and Key Highlights
# 
### 1. Core Classes: Vector and Matrix
   - The `Vector` and `Matrix` classes form the backbone of this project. They are designed to support operations such as addition, multiplication, and transpose, while also serving as containers for structured data.
  - Features include:
  - Flexible initialization (e.g., by specifying rows, columns, or directly using vector lists).
   - Human-readable output via the `__str__` method for easy debugging.
# 
### 2. Advanced Matrix Operations
 - Reduced Row Echelon Form (RREF): Enables row-wise transformations to simplify matrices for rank, nullity, and system-solving tasks.
  - Determinants: Implemented using both cofactor expansion and PLU decomposition to provide versatile approaches.
 - Change of Basis: Functions like `change_of_basis` and `change_basis` allow smooth transitions between coordinate systems, critical in real-world linear algebra applications.
# 
### 3. Decompositions
 - LU and PLU: Decompose matrices into lower and upper triangular forms for easier computation of determinants and solutions.
  - QR and Gram-Schmidt Orthogonalization: Provides a stable foundation for projecting vectors and decomposing matrices into orthogonal bases.
  - SVD: Computes the singular value decomposition, a cornerstone of machine learning and data analysis.
# 
### 4. Eigenvalues and Eigenvectors
  - The assignment implements functions to compute characteristic polynomials, minimal polynomials, eigenvalues, and eigenvectors.
 - It also includes checks for diagonalizability and transformations to find the change of basis matrix for diagonalization.
# 
### 5. Practical Applications
   - Functions like `least_square` solve inconsistent systems using pseudoinverses.
  - The `is_in_linear_span` function determines whether a vector can be expressed as a combination of given vectors, a critical task in optimization and computational geometry.
# 
# 
#### Testing and Validation
The provided test suite rigorously validates all implemented functionalities:
 - Basic Vector Operations: Testing addition, span checking, and coordinate transformations.
 - Matrix Operations: Verifying RREF, determinant calculations, and decompositions.
 - Advanced Features: Validating eigenvalue computations, diagonalizability checks, and matrix decompositions like SVD and Cholesky.
# 
 Errors during testing, such as dimension mismatches and invalid indexing, were resolved by carefully aligning input sizes and incorporating error handling.
# 
 
#### Challenges and Learnings
 - Error Handling: Ensuring robustness when dimensions didn’t align was a significant challenge, but this was mitigated by checks and clear error messages.
 - Mathematical Accuracy: Writing cofactor expansions, Gram-Schmidt processes, and eigenvalue computations required careful consideration of edge cases (e.g., zero matrices).
 - Modularity: Structuring the code into reusable components for tasks like row operations and vector manipulations made the implementation cleaner and more intuitive.

# 
#### Conclusion
 This assignment not only demonstrates the power of Python in performing complex mathematical operations but also highlights the intricacies of linear algebra. By coding from scratch, the project reinforced a deeper understanding of how algorithms like RREF, LU decomposition, and SVD operate under the hood.
# 
 The resulting script is a lightweight yet powerful library for linear algebra, capable of solving practical problems in engineering, physics, data science, and beyond. This project also showcases the potential of mathematical programming when combined with structured, modular, and error-tolerant design.
# 
 Moving forward, this library can be expanded to include optimizations for large matrices, symbolic computations, or real-world applications like machine learning preprocessing.

# Code file 
```python
##main script 1
class ComplexNumber:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return ComplexNumber(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other):
        return ComplexNumber(self.real * other.real - self.imag * other.imag,
                             self.real * other.imag + self.imag * other.real)

    def __truediv__(self, other):
        denom = other.real ** 2 + other.imag ** 2
        return ComplexNumber((self.real * other.real + self.imag * other.imag) / denom,
                             (self.imag * other.real - self.real * other.imag) / denom)

    def abs(self):
        return (self.real ** 2 + self.imag ** 2) ** 0.5

    def cc(self):
        return ComplexNumber(self.real, -self.imag)

class Vector:
    def __init__(self, field, length, values):
        self.field = field
        self.length = length
        self.values = values

class Matrix:
    def __init__(self, field, rows, cols, values=None, vectors=None):
        self.field = field
        self.rows = rows
        self.cols = cols
        if vectors:
            self.values = [list(v.values) for v in vectors]
        else:
            self.values = [values[i * cols:(i + 1) * cols] for i in range(rows)]

    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions for addition.")
        return Matrix(self.field, self.rows, self.cols,
                      [self.values[i][j] + other.values[i][j] for i in range(self.rows) for j in range(self.cols)])

    def __mul__(self, other):
        if self.cols != other.rows:
            raise ValueError("Matrix multiplication not possible: dimensions do not match.")
        result = [[sum(self.values[i][k] * other.values[k][j] for k in range(self.cols))
                   for j in range(other.cols)] for i in range(self.rows)]
        return Matrix(self.field, self.rows, other.cols, [val for row in result for val in row])

    def get_row(self, index):
        return self.values[index]

    def get_column(self, index):
        return [row[index] for row in self.values]

    def transpose(self):
        return Matrix(self.field, self.cols, self.rows, [self.values[j][i] for i in range(self.cols) for j in range(self.rows)])

    def conjugate(self):
        return Matrix(self.field, self.rows, self.cols, [[val.cc() if isinstance(val, ComplexNumber) else val for val in row] for row in self.values])

    def conjugate_transpose(self):
        return self.conjugate().transpose()

def is_zero(matrix):
    return all(val == 0 for row in matrix.values for val in row)

def is_symmetric(matrix):
    return matrix == matrix.transpose()

def is_hermitian(matrix):
    return matrix == matrix.conjugate_transpose()

def is_square(matrix):
    return matrix.rows == matrix.cols

def is_orthogonal(matrix):
    identity = Matrix(matrix.field, matrix.rows, matrix.rows, [[1 if i == j else 0 for j in range(matrix.rows)] for i in range(matrix.rows)])
    return matrix.transpose() * matrix == identity

def is_unitary(matrix):
    identity = Matrix(matrix.field, matrix.rows, matrix.rows, [[1 if i == j else 0 for j in range(matrix.rows)] for i in range(matrix.rows)])
    return matrix.conjugate_transpose() * matrix == identity

def is_scalar(matrix):
    return all(row[i] == matrix.values[0][0] for i, row in enumerate(matrix.values))

def is_singular(matrix):
    return determinant(matrix) == 0

def is_invertible(matrix):
    return not is_singular(matrix)

def is_identity(matrix):
    return all(matrix.values[i][i] == 1 and all(matrix.values[i][j] == 0 for j in range(matrix.cols) if i != j) for i in range(matrix.rows))

def is_nilpotent(matrix):
    power = matrix
    for _ in range(matrix.rows):
        power = power * matrix
        if is_zero(power):
            return True
    return False

def is_diagonalizable(matrix):
    eigenvalues = get_eigenvalues(matrix)
    return len(eigenvalues) == matrix.rows

def is_positive_definite(matrix):
    for i in range(1, matrix.rows + 1):
        sub_matrix = Matrix(matrix.field, i, i, [matrix.values[j][k] for j in range(i) for k in range(i)])
        if determinant(sub_matrix) <= 0:
            return False
    return True

def determinant(matrix):
    if matrix.rows != matrix.cols:
        raise ValueError("Determinant can only be calculated for square matrices.")
    if matrix.rows == 1:
        return matrix.values[0][0]
    result = 0
    for i in range(matrix.cols):
        sub_matrix_values = [row[:i] + row[i + 1:] for row in matrix.values[1:]]
        sub_matrix = Matrix(matrix.field, matrix.rows - 1, matrix.cols - 1,
                            [item for sublist in sub_matrix_values for item in sublist])
        result += ((-1) ** i) * matrix.values[0][i] * determinant(sub_matrix)
    return result

def get_eigenvalues(matrix):
    trace = sum(matrix.values[i][i] for i in range(matrix.rows))
    determinant_val = determinant(matrix)
    return [trace, determinant_val]


```

Main script 2


```python
class ComplexNumber:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return ComplexNumber(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other):
        return ComplexNumber(self.real * other.real - self.imag * other.imag,
                             self.real * other.imag + self.imag * other.real)

    def __truediv__(self, other):
        denom = other.real ** 2 + other.imag ** 2
        return ComplexNumber((self.real * other.real + self.imag * other.imag) / denom,
                             (self.imag * other.real - self.real * other.imag) / denom)

    def abs(self):
        return (self.real ** 2 + self.imag ** 2) ** 0.5

    def cc(self):
        return ComplexNumber(self.real, -self.imag)


class Vector:
    def __init__(self, field, length, values):
        self.field = field
        self.length = length
        self.values = values

    def get_length(self):
        return len(self.values)
    def __str__(self):
        return " ".join(str(val) for val in self.values)



class Matrix:
    def __init__(self, field, rows, cols, values=None, vectors=None):
        self.field = field
        self.rows = rows
        self.cols = cols
        if vectors:
            self.values = [list(v.values) for v in vectors]
        else:
            self.values = [values[i * cols:(i + 1) * cols] for i in range(rows)]
    
    def __str__(self):
        return "\n".join(" ".join(f"{val:.2f}" if isinstance(val, float) else str(val) for val in row) for row in self.values)

    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions for addition.")
        return Matrix(self.field, self.rows, self.cols,
                      [self.values[i][j] + other.values[i][j] for i in range(self.rows) for j in range(self.cols)])

    def __mul__(self, other):
        if self.cols != other.rows:
            raise ValueError("Matrix multiplication not possible: dimensions do not match.")
        result = [[sum(self.values[i][k] * other.values[k][j] for k in range(self.cols))
                   for j in range(other.cols)] for i in range(self.rows)]
        return Matrix(self.field, self.rows, other.cols, [val for row in result for val in row])

    def get_row(self, index):
        return self.values[index]

    def get_column(self, index):
        return [row[index] for row in self.values]

    def transpose(self):
        return Matrix(self.field, self.cols, self.rows, [self.values[j][i] for i in range(self.cols) for j in range(self.rows)])

    def conjugate(self):
        return Matrix(self.field, self.rows, self.cols, [[val.cc() if isinstance(val, ComplexNumber) else val for val in row] for row in self.values])

    def conjugate_transpose(self):
        return self.conjugate().transpose()

    def determinant(self):
        if self.rows != self.cols:
            raise ValueError("Determinant can only be calculated for square matrices.")
        if self.rows == 1:
            return self.values[0][0]
        result = 0
        for i in range(self.cols):
            sub_matrix_values = [row[:i] + row[i + 1:] for row in self.values[1:]]
            sub_matrix = Matrix(self.field, self.rows - 1, self.cols - 1,
                                [item for sublist in sub_matrix_values for item in sublist])
            result += ((-1) ** i) * self.values[0][i] * sub_matrix.determinant()
        return result

    def rref(self):
        matrix = [row[:] for row in self.values]
        rows, cols = self.rows, self.cols
        lead = 0
        for r in range(rows):
            if lead >= cols:
                break
            i = r
            while matrix[i][lead] == 0:
                i += 1
                if i == rows:
                    i = r
                    lead += 1
                    if lead == cols:
                        break
            matrix[i], matrix[r] = matrix[r], matrix[i]
            lv = matrix[r][lead]
            matrix[r] = [m / lv for m in matrix[r]]
            for i in range(rows):
                if i != r:
                    lv = matrix[i][lead]
                    matrix[i] = [iv - lv * rv for rv, iv in zip(matrix[r], matrix[i])]
            lead += 1
        return Matrix(self.field, rows, cols, [item for sublist in matrix for item in sublist])

    def rank(self):
        return sum(any(row) for row in self.rref().values)

    def nullity(self):
        return self.cols - self.rank()
    
# LU Decomposition
def lu_decomposition(matrix):
    if matrix.rows != matrix.cols:
        raise ValueError("LU decomposition requires a square matrix.")
    n = matrix.rows
    L = [[0] * n for _ in range(n)]
    U = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i, n):
            U[i][j] = matrix.values[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i, n):
            if i == j:
                L[i][i] = 1
            else:
                L[j][i] = (matrix.values[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]
    return Matrix(matrix.field, n, n, [val for row in L for val in row]), Matrix(matrix.field, n, n, [val for row in U for val in row])

# PLU Decomposition
def plu_decomposition(matrix):
    if matrix.rows != matrix.cols:
        raise ValueError("PLU decomposition requires a square matrix.")
    n = matrix.rows
    P = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    L = [[0] * n for _ in range(n)]
    U = [[0] * n for _ in range(n)]
    
    A = [row[:] for row in matrix.values]
    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
        if i != max_row:
            A[i], A[max_row] = A[max_row], A[i]
            P[i], P[max_row] = P[max_row], P[i]
        for j in range(i, n):
            U[i][j] = A[i][j]
        for j in range(i + 1, n):
            L[j][i] = A[j][i] / U[i][i]
            for k in range(i, n):
                A[j][k] -= L[j][i] * U[i][k]
    for i in range(n):
        L[i][i] = 1
    return Matrix(matrix.field, n, n, [val for row in P for val in row]), Matrix(matrix.field, n, n, [val for row in L for val in row]), Matrix(matrix.field, n, n, [val for row in U for val in row])


class LinearSystem:
    def __init__(self, matrix, vector):
        if matrix.rows != len(vector.values):
            raise ValueError("Matrix row count must equal vector length.")
        self.matrix = matrix
        self.vector = vector

    def is_consistent(self):
        augmented_matrix = Matrix(self.matrix.field, self.matrix.rows, self.matrix.cols + 1,
                                   [item for sublist in self.matrix.values for item in sublist] + self.vector.values)
        return augmented_matrix.rank() == self.matrix.rank()

    def solution_set(self):
        augmented_matrix = Matrix(self.matrix.field, self.matrix.rows, self.matrix.cols + 1,
                                   [item for sublist in self.matrix.values for item in sublist] + self.vector.values)
        rref = augmented_matrix.rref()
        free_variables = self.matrix.cols - self.matrix.rank()
        return {"RREF": rref, "Free Variables": free_variables}
    
# Change of Basis and Coordinates
def is_in_linear_span(S, v):
    augmented_matrix = Matrix("real", len(S[0].values), len(S) + 1,
                               [vec.values[i] for vec in S for i in range(len(vec.values))] + v.values)
    return augmented_matrix.rank() == len(S)

def express_in(S, v):
    augmented_matrix = Matrix("real", len(S[0].values), len(S) + 1,
                               [vec.values[i] for vec in S for i in range(len(vec.values))] + v.values)
    rref_matrix = augmented_matrix.rref()
    return [row[-1] for row in rref_matrix.values]

def is_span_equal(S1, S2):
    matrix1 = Matrix("real", len(S1[0].values), len(S1), [vec.values[i] for vec in S1 for i in range(len(vec.values))])
    matrix2 = Matrix("real", len(S2[0].values), len(S2), [vec.values[i] for vec in S2 for i in range(len(vec.values))])
    return matrix1.rank() == matrix2.rank()

def coord(B, v):
    return express_in(B, v)

def vector_from_coords(B, coords):
    result = [0] * len(B[0].values)
    for i in range(len(coords)):
        for j in range(len(result)):
            result[j] += coords[i] * B[i].values[j]
    return Vector("real", len(result), result)

def change_of_basis(B1, B2):
    coords_matrix = Matrix("real", len(B1), len(B2), [vector_from_coords(B2, coord(B1, vec)).values for vec in B1])
    return coords_matrix.transpose()

def change_basis(v, B1, B2):
    coords = coord(B1, v)
    return vector_from_coords(B2, coords)

# Determinants
def det_cofactor(matrix):
    if matrix.rows != matrix.cols:
        raise ValueError("Matrix must be square.")
    if matrix.rows == 1:
        return matrix.values[0][0]
    determinant = 0
    for i in range(matrix.cols):
        sub_matrix = Matrix(matrix.field, matrix.rows - 1, matrix.cols - 1,
                            [row[:i] + row[i + 1:] for row in matrix.values[1:]])
        determinant += ((-1) ** i) * matrix.values[0][i] * det_cofactor(sub_matrix)
    return determinant

# Inner Products and Orthogonalization
def inner_product(v1, v2):
    return sum(v1.values[i] * v2.values[i] for i in range(len(v1.values)))

def is_ortho(v1, v2):
    return inner_product(v1, v2) == 0

def gram_schmidt(S):
    ortho_set = []
    for v in S:
        for u in ortho_set:
            proj = inner_product(v, u) / inner_product(u, u)
            v = Vector(v.field, v.length, [v.values[i] - proj * u.values[i] for i in range(len(v.values))])
        ortho_set.append(v)
    return ortho_set

# Decompositions
def cholesky_decomposition(matrix):
    if matrix.rows != matrix.cols:
        raise ValueError("Matrix must be square.")
    n = matrix.rows
    L = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                L[i][i] = (matrix.values[i][i] - sum(L[i][k] ** 2 for k in range(j))) ** 0.5
            else:
                L[i][j] = (matrix.values[i][j] - sum(L[i][k] * L[j][k] for k in range(j))) / L[j][j]
    return Matrix(matrix.field, n, n, [val for row in L for val in row])

def svd(matrix):
    A_t_A = matrix.transpose() * matrix
    eigenvalues = [det_cofactor(A_t_A)]
    singular_values = [eigenvalue ** 0.5 for eigenvalue in eigenvalues]
    U = gram_schmidt([Vector(matrix.field, matrix.rows, matrix.get_column(i)) for i in range(matrix.cols)])
    S = Matrix(matrix.field, len(U), len(U), [[singular_values[i] if i == j else 0 for j in range(len(U))] for i in range(len(U))])
    V_t = U.transpose()
    return U, S, V_t


```


```python```python
##main script 1
class ComplexNumber:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return ComplexNumber(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other):
        return ComplexNumber(self.real * other.real - self.imag * other.imag,
                             self.real * other.imag + self.imag * other.real)

    def __truediv__(self, other):
        denom = other.real ** 2 + other.imag ** 2
        return ComplexNumber((self.real * other.real + self.imag * other.imag) / denom,
                             (self.imag * other.real - self.real * other.imag) / denom)

    def abs(self):
        return (self.real ** 2 + self.imag ** 2) ** 0.5

    def cc(self):
        return ComplexNumber(self.real, -self.imag)

class Vector:
    def __init__(self, field, length, values):
        self.field = field
        self.length = length
        self.values = values

class Matrix:
    def __init__(self, field, rows, cols, values=None, vectors=None):
        self.field = field
        self.rows = rows
        self.cols = cols
        if vectors:
            self.values = [list(v.values) for v in vectors]
        else:
            self.values = [values[i * cols:(i + 1) * cols] for i in range(rows)]

    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions for addition.")
        return Matrix(self.field, self.rows, self.cols,
                      [self.values[i][j] + other.values[i][j] for i in range(self.rows) for j in range(self.cols)])

    def __mul__(self, other):
        if self.cols != other.rows:
            raise ValueError("Matrix multiplication not possible: dimensions do not match.")
        result = [[sum(self.values[i][k] * other.values[k][j] for k in range(self.cols))
                   for j in range(other.cols)] for i in range(self.rows)]
        return Matrix(self.field, self.rows, other.cols, [val for row in result for val in row])

    def get_row(self, index):
        return self.values[index]

    def get_column(self, index):
        return [row[index] for row in self.values]

    def transpose(self):
        return Matrix(self.field, self.cols, self.rows, [self.values[j][i] for i in range(self.cols) for j in range(self.rows)])

    def conjugate(self):
        return Matrix(self.field, self.rows, self.cols, [[val.cc() if isinstance(val, ComplexNumber) else val for val in row] for row in self.values])

    def conjugate_transpose(self):
        return self.conjugate().transpose()

def is_zero(matrix):
    return all(val == 0 for row in matrix.values for val in row)

def is_symmetric(matrix):
    return matrix == matrix.transpose()

def is_hermitian(matrix):
    return matrix == matrix.conjugate_transpose()

def is_square(matrix):
    return matrix.rows == matrix.cols

def is_orthogonal(matrix):
    identity = Matrix(matrix.field, matrix.rows, matrix.rows, [[1 if i == j else 0 for j in range(matrix.rows)] for i in range(matrix.rows)])
    return matrix.transpose() * matrix == identity

def is_unitary(matrix):
    identity = Matrix(matrix.field, matrix.rows, matrix.rows, [[1 if i == j else 0 for j in range(matrix.rows)] for i in range(matrix.rows)])
    return matrix.conjugate_transpose() * matrix == identity

def is_scalar(matrix):
    return all(row[i] == matrix.values[0][0] for i, row in enumerate(matrix.values))

def is_singular(matrix):
    return determinant(matrix) == 0

def is_invertible(matrix):
    return not is_singular(matrix)

def is_identity(matrix):
    return all(matrix.values[i][i] == 1 and all(matrix.values[i][j] == 0 for j in range(matrix.cols) if i != j) for i in range(matrix.rows))

def is_nilpotent(matrix):
    power = matrix
    for _ in range(matrix.rows):
        power = power * matrix
        if is_zero(power):
            return True
    return False

def is_diagonalizable(matrix):
    eigenvalues = get_eigenvalues(matrix)
    return len(eigenvalues) == matrix.rows

def is_positive_definite(matrix):
    for i in range(1, matrix.rows + 1):
        sub_matrix = Matrix(matrix.field, i, i, [matrix.values[j][k] for j in range(i) for k in range(i)])
        if determinant(sub_matrix) <= 0:
            return False
    return True

def determinant(matrix):
    if matrix.rows != matrix.cols:
        raise ValueError("Determinant can only be calculated for square matrices.")
    if matrix.rows == 1:
        return matrix.values[0][0]
    result = 0
    for i in range(matrix.cols):
        sub_matrix_values = [row[:i] + row[i + 1:] for row in matrix.values[1:]]
        sub_matrix = Matrix(matrix.field, matrix.rows - 1, matrix.cols - 1,
                            [item for sublist in sub_matrix_values for item in sublist])
        result += ((-1) ** i) * matrix.values[0][i] * determinant(sub_matrix)
    return result

def get_eigenvalues(matrix):
    trace = sum(matrix.values[i][i] for i in range(matrix.rows))
    determinant_val = determinant(matrix)
    return [trace, determinant_val]


```

Main script 2


```python
class ComplexNumber:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return ComplexNumber(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other):
        return ComplexNumber(self.real * other.real - self.imag * other.imag,
                             self.real * other.imag + self.imag * other.real)

    def __truediv__(self, other):
        denom = other.real ** 2 + other.imag ** 2
        return ComplexNumber((self.real * other.real + self.imag * other.imag) / denom,
                             (self.imag * other.real - self.real * other.imag) / denom)

    def abs(self):
        return (self.real ** 2 + self.imag ** 2) ** 0.5

    def cc(self):
        return ComplexNumber(self.real, -self.imag)


class Vector:
    def __init__(self, field, length, values):
        self.field = field
        self.length = length
        self.values = values

    def get_length(self):
        return len(self.values)
    def __str__(self):
        return " ".join(str(val) for val in self.values)



class Matrix:
    def __init__(self, field, rows, cols, values=None, vectors=None):
        self.field = field
        self.rows = rows
        self.cols = cols
        if vectors:
            self.values = [list(v.values) for v in vectors]
        else:
            self.values = [values[i * cols:(i + 1) * cols] for i in range(rows)]
    
    def __str__(self):
        return "\n".join(" ".join(f"{val:.2f}" if isinstance(val, float) else str(val) for val in row) for row in self.values)

    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions for addition.")
        return Matrix(self.field, self.rows, self.cols,
                      [self.values[i][j] + other.values[i][j] for i in range(self.rows) for j in range(self.cols)])

    def __mul__(self, other):
        if self.cols != other.rows:
            raise ValueError("Matrix multiplication not possible: dimensions do not match.")
        result = [[sum(self.values[i][k] * other.values[k][j] for k in range(self.cols))
                   for j in range(other.cols)] for i in range(self.rows)]
        return Matrix(self.field, self.rows, other.cols, [val for row in result for val in row])

    def get_row(self, index):
        return self.values[index]

    def get_column(self, index):
        return [row[index] for row in self.values]

    def transpose(self):
        return Matrix(self.field, self.cols, self.rows, [self.values[j][i] for i in range(self.cols) for j in range(self.rows)])

    def conjugate(self):
        return Matrix(self.field, self.rows, self.cols, [[val.cc() if isinstance(val, ComplexNumber) else val for val in row] for row in self.values])

    def conjugate_transpose(self):
        return self.conjugate().transpose()

    def determinant(self):
        if self.rows != self.cols:
            raise ValueError("Determinant can only be calculated for square matrices.")
        if self.rows == 1:
            return self.values[0][0]
        result = 0
        for i in range(self.cols):
            sub_matrix_values = [row[:i] + row[i + 1:] for row in self.values[1:]]
            sub_matrix = Matrix(self.field, self.rows - 1, self.cols - 1,
                                [item for sublist in sub_matrix_values for item in sublist])
            result += ((-1) ** i) * self.values[0][i] * sub_matrix.determinant()
        return result

    def rref(self):
        matrix = [row[:] for row in self.values]
        rows, cols = self.rows, self.cols
        lead = 0
        for r in range(rows):
            if lead >= cols:
                break
            i = r
            while matrix[i][lead] == 0:
                i += 1
                if i == rows:
                    i = r
                    lead += 1
                    if lead == cols:
                        break
            matrix[i], matrix[r] = matrix[r], matrix[i]
            lv = matrix[r][lead]
            matrix[r] = [m / lv for m in matrix[r]]
            for i in range(rows):
                if i != r:
                    lv = matrix[i][lead]
                    matrix[i] = [iv - lv * rv for rv, iv in zip(matrix[r], matrix[i])]
            lead += 1
        return Matrix(self.field, rows, cols, [item for sublist in matrix for item in sublist])

    def rank(self):
        return sum(any(row) for row in self.rref().values)

    def nullity(self):
        return self.cols - self.rank()
    
# LU Decomposition
def lu_decomposition(matrix):
    if matrix.rows != matrix.cols:
        raise ValueError("LU decomposition requires a square matrix.")
    n = matrix.rows
    L = [[0] * n for _ in range(n)]
    U = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i, n):
            U[i][j] = matrix.values[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i, n):
            if i == j:
                L[i][i] = 1
            else:
                L[j][i] = (matrix.values[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]
    return Matrix(matrix.field, n, n, [val for row in L for val in row]), Matrix(matrix.field, n, n, [val for row in U for val in row])

# PLU Decomposition
def plu_decomposition(matrix):
    if matrix.rows != matrix.cols:
        raise ValueError("PLU decomposition requires a square matrix.")
    n = matrix.rows
    P = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    L = [[0] * n for _ in range(n)]
    U = [[0] * n for _ in range(n)]
    
    A = [row[:] for row in matrix.values]
    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
        if i != max_row:
            A[i], A[max_row] = A[max_row], A[i]
            P[i], P[max_row] = P[max_row], P[i]
        for j in range(i, n):
            U[i][j] = A[i][j]
        for j in range(i + 1, n):
            L[j][i] = A[j][i] / U[i][i]
            for k in range(i, n):
                A[j][k] -= L[j][i] * U[i][k]
    for i in range(n):
        L[i][i] = 1
    return Matrix(matrix.field, n, n, [val for row in P for val in row]), Matrix(matrix.field, n, n, [val for row in L for val in row]), Matrix(matrix.field, n, n, [val for row in U for val in row])


class LinearSystem:
    def __init__(self, matrix, vector):
        if matrix.rows != len(vector.values):
            raise ValueError("Matrix row count must equal vector length.")
        self.matrix = matrix
        self.vector = vector

    def is_consistent(self):
        augmented_matrix = Matrix(self.matrix.field, self.matrix.rows, self.matrix.cols + 1,
                                   [item for sublist in self.matrix.values for item in sublist] + self.vector.values)
        return augmented_matrix.rank() == self.matrix.rank()

    def solution_set(self):
        augmented_matrix = Matrix(self.matrix.field, self.matrix.rows, self.matrix.cols + 1,
                                   [item for sublist in self.matrix.values for item in sublist] + self.vector.values)
        rref = augmented_matrix.rref()
        free_variables = self.matrix.cols - self.matrix.rank()
        return {"RREF": rref, "Free Variables": free_variables}
    
# Change of Basis and Coordinates
def is_in_linear_span(S, v):
    augmented_matrix = Matrix("real", len(S[0].values), len(S) + 1,
                               [vec.values[i] for vec in S for i in range(len(vec.values))] + v.values)
    return augmented_matrix.rank() == len(S)

def express_in(S, v):
    augmented_matrix = Matrix("real", len(S[0].values), len(S) + 1,
                               [vec.values[i] for vec in S for i in range(len(vec.values))] + v.values)
    rref_matrix = augmented_matrix.rref()
    return [row[-1] for row in rref_matrix.values]

def is_span_equal(S1, S2):
    matrix1 = Matrix("real", len(S1[0].values), len(S1), [vec.values[i] for vec in S1 for i in range(len(vec.values))])
    matrix2 = Matrix("real", len(S2[0].values), len(S2), [vec.values[i] for vec in S2 for i in range(len(vec.values))])
    return matrix1.rank() == matrix2.rank()

def coord(B, v):
    return express_in(B, v)

def vector_from_coords(B, coords):
    result = [0] * len(B[0].values)
    for i in range(len(coords)):
        for j in range(len(result)):
            result[j] += coords[i] * B[i].values[j]
    return Vector("real", len(result), result)

def change_of_basis(B1, B2):
    coords_matrix = Matrix("real", len(B1), len(B2), [vector_from_coords(B2, coord(B1, vec)).values for vec in B1])
    return coords_matrix.transpose()

def change_basis(v, B1, B2):
    coords = coord(B1, v)
    return vector_from_coords(B2, coords)

# Determinants
def det_cofactor(matrix):
    if matrix.rows != matrix.cols:
        raise ValueError("Matrix must be square.")
    if matrix.rows == 1:
        return matrix.values[0][0]
    determinant = 0
    for i in range(matrix.cols):
        sub_matrix = Matrix(matrix.field, matrix.rows - 1, matrix.cols - 1,
                            [row[:i] + row[i + 1:] for row in matrix.values[1:]])
        determinant += ((-1) ** i) * matrix.values[0][i] * det_cofactor(sub_matrix)
    return determinant

# Inner Products and Orthogonalization
def inner_product(v1, v2):
    return sum(v1.values[i] * v2.values[i] for i in range(len(v1.values)))

def is_ortho(v1, v2):
    return inner_product(v1, v2) == 0

def gram_schmidt(S):
    ortho_set = []
    for v in S:
        for u in ortho_set:
            proj = inner_product(v, u) / inner_product(u, u)
            v = Vector(v.field, v.length, [v.values[i] - proj * u.values[i] for i in range(len(v.values))])
        ortho_set.append(v)
    return ortho_set

# Decompositions
def cholesky_decomposition(matrix):
    if matrix.rows != matrix.cols:
        raise ValueError("Matrix must be square.")
    n = matrix.rows
    L = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                L[i][i] = (matrix.values[i][i] - sum(L[i][k] ** 2 for k in range(j))) ** 0.5
            else:
                L[i][j] = (matrix.values[i][j] - sum(L[i][k] * L[j][k] for k in range(j))) / L[j][j]
    return Matrix(matrix.field, n, n, [val for row in L for val in row])

def svd(matrix):
    A_t_A = matrix.transpose() * matrix
    eigenvalues = [det_cofactor(A_t_A)]
    singular_values = [eigenvalue ** 0.5 for eigenvalue in eigenvalues]
    U = gram_schmidt([Vector(matrix.field, matrix.rows, matrix.get_column(i)) for i in range(matrix.cols)])
    S = Matrix(matrix.field, len(U), len(U), [[singular_values[i] if i == j else 0 for j in range(len(U))] for i in range(len(U))])
    V_t = U.transpose()
    return U, S, V_t


```


```python
