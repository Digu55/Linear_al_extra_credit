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
