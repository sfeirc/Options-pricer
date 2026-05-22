#pragma once
#include <algorithm>
#include <cmath>

namespace options {

// ---------------------------------------------------------------------------
// Crank-Nicolson finite difference pricer for American options
// Template parameters allow compile-time sizing (zero heap allocation)
// N_S: number of spatial nodes, N_T: number of time steps
// ---------------------------------------------------------------------------
template<int N_S = 200, int N_T = 100>
class FDAmericanPricer {
public:
    // All arrays on the stack; aligned for SIMD / cache-line efficiency
    alignas(64) double V_curr[N_S];
    alignas(64) double V_prev[N_S];
    alignas(64) double S_grid[N_S];
    alignas(64) double intrinsic[N_S];
    alignas(64) double a[N_S];        // lower diagonal
    alignas(64) double b[N_S];        // main diagonal
    alignas(64) double c[N_S];        // upper diagonal
    alignas(64) double d[N_S];        // RHS
    alignas(64) double c_prime[N_S];  // modified upper diagonal (Thomas)
    alignas(64) double d_prime[N_S];  // modified RHS / solution (Thomas)

    double price(double S0, double K, double T, double r, double sigma, bool is_call) noexcept {
        const double S_max = 4.0 * S0;
        const double dt    = T / static_cast<double>(N_T);
        const double eta   = 3.0;

        // Non-uniform spatial grid: concentration near S0 (ATM region)
        // S_grid[i] = S_max * sinh(eta * i/(N_S-1)) / sinh(eta)
        const double sinh_eta = std::sinh(eta);
        for (int i = 0; i < N_S; ++i) {
            S_grid[i] = S_max * std::sinh(eta * static_cast<double>(i) / (N_S - 1)) / sinh_eta;
        }

        // Terminal condition (payoff at expiry)
        for (int i = 0; i < N_S; ++i) {
            V_curr[i]    = is_call ? std::max(S_grid[i] - K, 0.0)
                                   : std::max(K - S_grid[i], 0.0);
            intrinsic[i] = V_curr[i];
        }

        // Backward time-stepping from T to 0
        for (int t = N_T - 1; t >= 0; --t) {
            // tau = time remaining at current step
            double tau = static_cast<double>(N_T - t) * dt;
            build_cn_system(dt, r, sigma, is_call, K, S_max, tau);
            thomas_solve();
            // d_prime now holds V_new; enforce early exercise
            for (int i = 0; i < N_S; ++i) {
                double ie = is_call ? std::max(S_grid[i] - K, 0.0)
                                    : std::max(K - S_grid[i], 0.0);
                V_curr[i] = std::max(d_prime[i], ie);
            }
        }

        return interpolate(S0);
    }

private:
    // Build the CN tridiagonal system for one time step
    void build_cn_system(double dt, double r, double sigma,
                         bool is_call, double K, double S_max, double tau) noexcept {
        // Boundary values
        double disc = std::exp(-r * tau);
        double bc_low  = is_call ? 0.0               : K * disc;   // i=0
        double bc_high = is_call ? S_max - K * disc  : 0.0;        // i=N_S-1

        // Interior nodes: i = 1 .. N_S-2
        for (int i = 1; i < N_S - 1; ++i) {
            double dS_m   = S_grid[i] - S_grid[i - 1];
            double dS_p   = S_grid[i + 1] - S_grid[i];
            double dS_avg = 0.5 * (dS_m + dS_p);

            double sig2S2  = sigma * sigma * S_grid[i] * S_grid[i];
            double alpha_i = 0.5 * dt * sig2S2 / (dS_m * dS_avg);
            double beta_i  = 0.5 * dt * r * S_grid[i] / (2.0 * dS_avg);

            // LHS (implicit part): coefficients for V_new
            a[i] = -0.5 * (alpha_i - beta_i);   // V_{i-1}^{n+1}
            b[i] =  1.0 + alpha_i + 0.5 * r * dt; // V_i^{n+1}
            c[i] = -0.5 * (alpha_i + beta_i);   // V_{i+1}^{n+1}

            // RHS (explicit part): from V_curr with opposite-sign CN coefficients
            double rhs_coeff_m =  0.5 * (alpha_i - beta_i);
            double rhs_coeff_0 =  1.0 - alpha_i - 0.5 * r * dt;
            double rhs_coeff_p =  0.5 * (alpha_i + beta_i);

            d[i] = rhs_coeff_m * V_curr[i - 1]
                 + rhs_coeff_0 * V_curr[i]
                 + rhs_coeff_p * V_curr[i + 1];
        }

        // Boundary conditions imposed directly
        // i = 0
        a[0] = 0.0;  b[0] = 1.0;  c[0] = 0.0;
        d[0] = bc_low;

        // i = N_S-1
        a[N_S - 1] = 0.0;  b[N_S - 1] = 1.0;  c[N_S - 1] = 0.0;
        d[N_S - 1] = bc_high;
    }

    // Thomas algorithm (TDMA) — forward sweep + back-substitution
    // Solution is written into d_prime[]
    void thomas_solve() noexcept {
        // Forward sweep
        c_prime[0] = c[0] / b[0];
        d_prime[0] = d[0] / b[0];

        for (int i = 1; i < N_S; ++i) {
            double m    = b[i] - a[i] * c_prime[i - 1];
            c_prime[i]  = c[i] / m;
            d_prime[i]  = (d[i] - a[i] * d_prime[i - 1]) / m;
        }

        // Back-substitution
        // d_prime[N_S-1] is already the solution at i=N_S-1
        for (int i = N_S - 2; i >= 0; --i) {
            d_prime[i] -= c_prime[i] * d_prime[i + 1];
        }
    }

    // Cubic (3-point quadratic) interpolation at S
    double interpolate(double S) const noexcept {
        // Find bracket
        int lo = 0, hi = N_S - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            if (S_grid[mid] <= S) lo = mid;
            else                  hi = mid;
        }

        if (lo == N_S - 2) lo = N_S - 3; // ensure 3-point window fits
        if (lo < 0)        lo = 0;

        // Quadratic Lagrange interpolation through nodes lo, lo+1, lo+2
        int i0 = lo, i1 = lo + 1, i2 = lo + 2;
        if (i2 >= N_S) { i2 = N_S - 1; i1 = i2 - 1; i0 = i2 - 2; }

        double x0 = S_grid[i0], x1 = S_grid[i1], x2 = S_grid[i2];
        double v0 = V_curr[i0], v1 = V_curr[i1], v2 = V_curr[i2];

        double L0 = (S - x1) * (S - x2) / ((x0 - x1) * (x0 - x2));
        double L1 = (S - x0) * (S - x2) / ((x1 - x0) * (x1 - x2));
        double L2 = (S - x0) * (S - x1) / ((x2 - x0) * (x2 - x1));

        double val = v0 * L0 + v1 * L1 + v2 * L2;
        return std::max(val, 0.0);
    }
};

} // namespace options
