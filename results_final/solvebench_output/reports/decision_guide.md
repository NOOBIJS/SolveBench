# SolveBench — Solver Decision Guide

_Generated 2026-08-23 06:57 from 930 real matrices x 10 solvers._

## Overall

- **Most reliable:** APK (ours) — solved 66% of all matrices
- **Most accurate:** Cholesky — median relative error 4.38e-14
- **Fastest:** Jacobi — median runtime 0.0086 s

| Method | Family | Success rate | Median rel. error | Median runtime (s) |
|---|---|---|---|---|
| Gauss elimination | direct | 65% | 2.36e-12 | 0.1841 |
| Gauss-Jordan | direct | 65% | 3.24e-12 | 0.3152 |
| LU | direct | 65% | 2.96e-12 | 0.1776 |
| Cholesky | direct | 7% | 4.38e-14 | 0.0168 |
| Jacobi | iterative | 8% | 8.13e-09 | 0.0086 |
| Gauss-Seidel | iterative | 13% | 1.07e-07 | 0.0788 |
| SOR | iterative | 13% | 8.14e-08 | 0.1460 |
| Conjugate Gradient | iterative | 10% | 1.62e-05 | 0.0219 |
| BiCGSTAB | iterative | 35% | 1.11e-06 | 0.0307 |
| APK (ours) | iterative | 66% | 6.03e-14 | 0.0090 |

## By domain

### 2D 3D Problem

- **Recommended:** `APK (ours)` (most accurate here, median relative error 2.25e-13)
- **Fastest:** `APK (ours)` (0.0050 s median)
- **Most reliable:** `APK (ours)` (80% success)
- **Avoid / use with care:** APK (ours), BiCGSTAB, Conjugate Gradient, Gauss-Seidel, Jacobi, SOR (failed to converge on at least one matrix here)

### Acoustics Problem

- **Recommended:** `Gauss-Jordan` (most accurate here, median relative error 6.31e-15)
- **Fastest:** `Jacobi` (0.0010 s median)
- **Most reliable:** `APK (ours)` (100% success)
- **Avoid / use with care:** Gauss-Seidel, Jacobi, SOR (failed to converge on at least one matrix here)

### Chemical Process Simulation Problem

- **Recommended:** `APK (ours)` (most accurate here, median relative error 4.48e-13)
- **Fastest:** `APK (ours)` (0.0026 s median)
- **Most reliable:** `Gauss elimination` (55% success)
- **Avoid / use with care:** APK (ours), BiCGSTAB, Gauss-Seidel, Jacobi, SOR (failed to converge on at least one matrix here)

### Circuit Simulation Problem

- **Recommended:** `APK (ours)` (most accurate here, median relative error 1.03e-08)
- **Fastest:** `APK (ours)` (0.0080 s median)
- **Most reliable:** `Gauss elimination` (96% success)
- **Avoid / use with care:** APK (ours), BiCGSTAB, Gauss-Seidel, Jacobi (failed to converge on at least one matrix here)

### Computational Chemistry Problem

- **Recommended:** `APK (ours)` (most accurate here, median relative error 1.44e-12)
- **Fastest:** `Gauss elimination` (0.4564 s median)
- **Most reliable:** `APK (ours)` (100% success)
- **Avoid / use with care:** BiCGSTAB, Gauss-Seidel, Jacobi, SOR (failed to converge on at least one matrix here)

### Computational Fluid Dynamics

- **Recommended:** `APK (ours)` (most accurate here, median relative error 2.35e-13)
- **Fastest:** `APK (ours)` (0.0418 s median)
- **Most reliable:** `APK (ours)` (100% success)

### Computational Fluid Dynamics Problem

- **Recommended:** `APK (ours)` (most accurate here, median relative error 2.54e-14)
- **Fastest:** `Jacobi` (0.0081 s median)
- **Most reliable:** `APK (ours)` (58% success)
- **Avoid / use with care:** APK (ours), BiCGSTAB, Conjugate Gradient, Gauss-Seidel, Jacobi, SOR (failed to converge on at least one matrix here)

### Computer Graphics Vision Problem

- **Recommended:** `Cholesky` (most accurate here, median relative error 9.75e-14)
- **Fastest:** `Conjugate Gradient` (0.0084 s median)
- **Most reliable:** `APK (ours)` (100% success)

### Counter Example Problem

- **Recommended:** `APK (ours)` (most accurate here, median relative error 3.42e-12)
- **Fastest:** `APK (ours)` (0.0149 s median)
- **Most reliable:** `APK (ours)` (75% success)
- **Avoid / use with care:** APK (ours), BiCGSTAB, Gauss-Seidel, Jacobi, SOR (failed to converge on at least one matrix here)

### Directed Weighted Graph

- **Recommended:** `APK (ours)` (most accurate here, median relative error 3.16e-16)
- **Fastest:** `BiCGSTAB` (0.0010 s median)
- **Most reliable:** `LU` (82% success)
- **Avoid / use with care:** APK (ours), BiCGSTAB, Gauss-Seidel, Jacobi, SOR (failed to converge on at least one matrix here)

### Economic Problem

- **Recommended:** `APK (ours)` (most accurate here, median relative error 1.64e-15)
- **Fastest:** `BiCGSTAB` (0.0033 s median)
- **Most reliable:** `APK (ours)` (31% success)
- **Avoid / use with care:** APK (ours), BiCGSTAB (failed to converge on at least one matrix here)

### Eigenvalue Model Reduction Problem

- **Recommended:** `Gauss-Jordan` (most accurate here, median relative error 5.57e-08)
- **Fastest:** `Gauss elimination` (0.0025 s median)
- **Most reliable:** `APK (ours)` (11% success)
- **Avoid / use with care:** APK (ours), BiCGSTAB (failed to converge on at least one matrix here)

### Electromagnetics Problem

- **Recommended:** `APK (ours)` (most accurate here, median relative error 3.42e-15)
- **Fastest:** `Jacobi` (0.0007 s median)
- **Most reliable:** `APK (ours)` (60% success)
- **Avoid / use with care:** APK (ours), BiCGSTAB, Conjugate Gradient, Gauss-Seidel, Jacobi, SOR (failed to converge on at least one matrix here)

### Linear Programming Problem

- **Recommended:** `APK (ours)` (most accurate here, median relative error 1.64e+05)
- **Fastest:** `BiCGSTAB` (0.0004 s median)
- **Most reliable:** `APK (ours)` (100% success)

### Materials Problem

- **Recommended:** `LU` (most accurate here, median relative error 2.14e-14)
- **Fastest:** `Jacobi` (0.0004 s median)
- **Most reliable:** `APK (ours)` (33% success)
- **Avoid / use with care:** APK (ours), BiCGSTAB, Conjugate Gradient, Gauss-Seidel, Jacobi, SOR (failed to converge on at least one matrix here)

### Model Reduction Problem

- **Recommended:** `APK (ours)` (most accurate here, median relative error 2.43e-14)
- **Fastest:** `Cholesky` (0.0004 s median)
- **Most reliable:** `APK (ours)` (69% success)
- **Avoid / use with care:** APK (ours), BiCGSTAB, Gauss-Seidel, Jacobi, SOR (failed to converge on at least one matrix here)

### Optimal Control Problem

- **Recommended:** `APK (ours)` (most accurate here, median relative error 6.30e-12)
- **Fastest:** `Conjugate Gradient` (0.0740 s median)
- **Most reliable:** `Gauss elimination` (51% success)
- **Avoid / use with care:** APK (ours), BiCGSTAB, Conjugate Gradient (failed to converge on at least one matrix here)

### Optimization Problem

- **Recommended:** `APK (ours)` (most accurate here, median relative error 1.00e-13)
- **Fastest:** `APK (ours)` (0.0085 s median)
- **Most reliable:** `APK (ours)` (56% success)
- **Avoid / use with care:** APK (ours), BiCGSTAB, Gauss-Seidel, Jacobi, SOR (failed to converge on at least one matrix here)

### Power Network Problem

- **Recommended:** `APK (ours)` (most accurate here, median relative error 3.55e-13)
- **Fastest:** `Conjugate Gradient` (0.0254 s median)
- **Most reliable:** `APK (ours)` (88% success)
- **Avoid / use with care:** APK (ours), BiCGSTAB, Gauss-Seidel, Jacobi, SOR (failed to converge on at least one matrix here)

### Random Matrices

- **Recommended:** `APK (ours)` (most accurate here, median relative error 7.82e-15)
- **Fastest:** `APK (ours)` (0.0016 s median)
- **Most reliable:** `APK (ours)` (100% success)
- **Avoid / use with care:** BiCGSTAB, Gauss-Seidel, Jacobi, SOR (failed to converge on at least one matrix here)

### Robotics Problem

- **Recommended:** `Gauss-Jordan` (most accurate here, median relative error 3.57e-11)
- **Fastest:** `LU` (0.0069 s median)
- **Most reliable:** `Gauss elimination` (100% success)
- **Avoid / use with care:** APK (ours), BiCGSTAB (failed to converge on at least one matrix here)

### Semiconductor Device Problem

- **Recommended:** `APK (ours)` (most accurate here, median relative error 1.26e-15)
- **Fastest:** `Jacobi` (0.0046 s median)
- **Most reliable:** `APK (ours)` (100% success)
- **Avoid / use with care:** Jacobi (failed to converge on at least one matrix here)

### Statistical Mathematical Problem

- **Recommended:** `APK (ours)` (most accurate here, median relative error 3.19e-12)
- **Fastest:** `BiCGSTAB` (0.0010 s median)
- **Most reliable:** `APK (ours)` (83% success)
- **Avoid / use with care:** APK (ours), BiCGSTAB, Gauss-Seidel, Jacobi, SOR (failed to converge on at least one matrix here)

### Structural Problem

- **Recommended:** `Jacobi` (most accurate here, median relative error 4.20e-17)
- **Fastest:** `Jacobi` (0.0005 s median)
- **Most reliable:** `APK (ours)` (90% success)
- **Avoid / use with care:** APK (ours), BiCGSTAB, Conjugate Gradient, Gauss-Seidel, Jacobi, SOR (failed to converge on at least one matrix here)

### Theoretical Quantum Chemistry Problem

- **Recommended:** `APK (ours)` (most accurate here, median relative error 5.32e-16)
- **Fastest:** `Jacobi` (0.0139 s median)
- **Most reliable:** `APK (ours)` (100% success)
- **Avoid / use with care:** Gauss-Seidel, Jacobi, SOR (failed to converge on at least one matrix here)

### Thermal Problem

- **Recommended:** `APK (ours)` (most accurate here, median relative error 9.66e-15)
- **Fastest:** `APK (ours)` (0.0245 s median)
- **Most reliable:** `APK (ours)` (100% success)
- **Avoid / use with care:** Gauss-Seidel, Jacobi, SOR (failed to converge on at least one matrix here)

### Undirected Weighted Graph

- **Recommended:** `APK (ours)` (most accurate here, median relative error 1.74e-14)
- **Fastest:** `APK (ours)` (0.0084 s median)
- **Most reliable:** `Gauss elimination` (68% success)
- **Avoid / use with care:** APK (ours), BiCGSTAB (failed to converge on at least one matrix here)
