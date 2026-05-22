#pragma once
#include <cmath>
#include <functional>
#include <cstring>
#include <algorithm>

namespace options {

// ---------------------------------------------------------------------------
// Levenberg-Marquardt nonlinear least squares optimizer
// Minimises ||f(x)||^2 where f: R^n -> R^m
// Designed for n <= 8 (SABR has n=4); uses fixed-size 8x8 system at most
// ---------------------------------------------------------------------------
class LevenbergMarquardt {
public:
    struct Result {
        bool   converged;
        int    iterations;
        double rms_error;
    };

    using ResidualFn = std::function<void(const double* x, double* f, int m)>;
    using JacobianFn = std::function<void(const double* x, double* J, int m, int n)>;

    static constexpr int MAX_N = 8;  // max parameter count

    Result optimise(
        double*      x,           // initial params (in/out), length n_params
        int          n_params,    // n
        int          n_residuals, // m
        ResidualFn   residual_fn,
        JacobianFn   jacobian_fn,
        int          max_iter = 100,
        double       tol      = 1e-8) noexcept
    {
        const int n = n_params;
        const int m = n_residuals;

        // Allocate working buffers on the stack (m up to 512 safely)
        // For larger m, caller should use heap — kept simple here
        constexpr int MAX_M = 512;
        double f[MAX_M]    = {};
        double J[MAX_M * MAX_N] = {};  // m x n, row-major
        double JtJ[MAX_N * MAX_N] = {};
        double Jtf[MAX_N] = {};
        double diag[MAX_N] = {};
        double dx[MAX_N] = {};
        double x_new[MAX_N] = {};
        double A[MAX_N * MAX_N] = {};

        double lambda = 1e-3;

        // Initial residuals
        residual_fn(x, f, m);
        double cost = dot(f, f, m);

        Result result{false, 0, 0.0};

        for (int iter = 0; iter < max_iter; ++iter) {
            result.iterations = iter + 1;

            // Compute Jacobian
            jacobian_fn(x, J, m, n);

            // Compute JtJ and Jt*f
            // JtJ[i,j] = sum_k J[k,i]*J[k,j]
            for (int i = 0; i < n; ++i) {
                for (int jj = 0; jj < n; ++jj) {
                    double s = 0.0;
                    for (int k = 0; k < m; ++k)
                        s += J[k * n + i] * J[k * n + jj];
                    JtJ[i * n + jj] = s;
                }
                double s = 0.0;
                for (int k = 0; k < m; ++k)
                    s += J[k * n + i] * f[k];
                Jtf[i] = s;
                diag[i] = JtJ[i * n + i];
            }

            // Build A = JtJ + lambda * diag(JtJ)
            for (int i = 0; i < n; ++i)
                for (int jj = 0; jj < n; ++jj)
                    A[i * n + jj] = JtJ[i * n + jj] + (i == jj ? lambda * diag[i] : 0.0);

            // Solve A * dx = -Jtf
            double neg_Jtf[MAX_N];
            for (int i = 0; i < n; ++i) neg_Jtf[i] = -Jtf[i];

            bool ok = (n == 4) ? solve_4x4(A, neg_Jtf, dx)
                                : solve_nxn(A, neg_Jtf, dx, n);
            if (!ok) {
                lambda *= 10.0;
                continue;
            }

            // Trial step
            for (int i = 0; i < n; ++i) x_new[i] = x[i] + dx[i];

            // Evaluate new cost
            residual_fn(x_new, f, m);
            double new_cost = dot(f, f, m);

            if (new_cost < cost) {
                // Accept step
                for (int i = 0; i < n; ++i) x[i] = x_new[i];
                cost = new_cost;
                lambda   *= 0.1;
                if (lambda < 1e-16) lambda = 1e-16;

                // Check convergence
                double dx_norm = 0.0;
                for (int i = 0; i < n; ++i) dx_norm += dx[i] * dx[i];
                if (std::sqrt(dx_norm) < tol && iter > 0) {
                    result.converged = true;
                    break;
                }

                // Recompute residuals at new x for next Jacobian
                residual_fn(x, f, m);
            } else {
                // Reject step
                lambda *= 10.0;
                if (lambda > 1e8) {
                    result.converged = false;
                    break;
                }
                // Restore f to current x
                residual_fn(x, f, m);
                cost = dot(f, f, m);
            }

            // Gradient norm convergence check
            double grad_norm = 0.0;
            for (int i = 0; i < n; ++i) grad_norm += Jtf[i] * Jtf[i];
            if (std::sqrt(grad_norm) < tol * tol) {
                result.converged = true;
                break;
            }
        }

        // Recompute final residuals
        residual_fn(x, f, m);
        double final_cost = dot(f, f, m);
        result.rms_error  = std::sqrt(final_cost / m);
        return result;
    }

    // Solve 4x4 system A*x = b via Gaussian elimination with partial pivoting
    // Returns false if matrix is singular
    bool solve_4x4(const double* A, const double* b, double* x) noexcept {
        return solve_nxn(A, b, x, 4);
    }

    // General nxn solver (n <= MAX_N)
    bool solve_nxn(const double* A, const double* b, double* x, int n) noexcept {
        // Augmented matrix [A | b], in-place
        double aug[MAX_N * (MAX_N + 1)] = {};
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j)
                aug[i * (n + 1) + j] = A[i * n + j];
            aug[i * (n + 1) + n] = b[i];
        }

        // Forward elimination with partial pivoting
        for (int col = 0; col < n; ++col) {
            // Find pivot
            int pivot = col;
            double max_val = std::abs(aug[col * (n + 1) + col]);
            for (int row = col + 1; row < n; ++row) {
                double val = std::abs(aug[row * (n + 1) + col]);
                if (val > max_val) { max_val = val; pivot = row; }
            }
            if (max_val < 1e-15) return false;  // singular

            // Swap rows
            if (pivot != col) {
                for (int j = 0; j <= n; ++j) {
                    double tmp = aug[col * (n + 1) + j];
                    aug[col   * (n + 1) + j] = aug[pivot * (n + 1) + j];
                    aug[pivot * (n + 1) + j] = tmp;
                }
            }

            double inv_pivot = 1.0 / aug[col * (n + 1) + col];
            for (int row = col + 1; row < n; ++row) {
                double factor = aug[row * (n + 1) + col] * inv_pivot;
                for (int j = col; j <= n; ++j)
                    aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
            }
        }

        // Back-substitution
        for (int i = n - 1; i >= 0; --i) {
            double s = aug[i * (n + 1) + n];
            for (int j = i + 1; j < n; ++j)
                s -= aug[i * (n + 1) + j] * x[j];
            x[i] = s / aug[i * (n + 1) + i];
        }
        return true;
    }

private:
    static double dot(const double* a, const double* b, int n) noexcept {
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += a[i] * b[i];
        return s;
    }
};

} // namespace options
