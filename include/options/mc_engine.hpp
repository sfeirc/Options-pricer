#pragma once
#include <cstdint>
#include <cmath>
#include <functional>
#include "sobol.hpp"
#include "bs_scalar.hpp"

namespace options {

struct MCResult {
    double price;            // plain MC estimate
    double std_error;        // standard error of plain MC
    double price_antithetic; // antithetic variate estimate
    double price_cv;         // control variate adjusted
    double price_full;       // all variance reductions combined
};

// Payoff function: receives full path, number of steps, and initial price
using Payoff = std::function<double(const double* path, int n_steps, double S0)>;

// ---------------------------------------------------------------------------
// Monte Carlo engine with four variance reduction techniques:
//   1. Antithetic variates
//   2. Control variate (geometric-average Asian for path-dependent)
//   3. Importance sampling (shift drift for OTM options)
//   4. Stratified sampling (via Sobol quasi-random)
// ---------------------------------------------------------------------------
class MCEngine {
public:
    struct Config {
        int  n_paths             = 100000;
        int  n_steps             = 252;
        bool antithetic          = true;
        bool control_variate     = true;
        bool importance_sampling = false;
        bool stratified          = true;  // use Sobol instead of pseudo-random
    };

    // Default Config object used when none is passed
    static const Config& default_config() noexcept {
        static const Config cfg{};
        return cfg;
    }

    MCResult price_european(double S, double K, double T, double r, double sigma,
                             bool is_call,
                             const Config& cfg = MCEngine::default_config()) noexcept;

    MCResult price_asian(double S, double K, double T, double r, double sigma,
                          bool is_call, bool geometric,
                          const Config& cfg = MCEngine::default_config()) noexcept;

    MCResult price_barrier(double S, double K, double H, double T, double r, double sigma,
                            bool is_call, bool up, bool is_out,
                            const Config& cfg = MCEngine::default_config()) noexcept;

private:
    SobolGenerator sobol_{1};

    // Generate GBM path using exact log-normal discretisation
    // S_{t+dt} = S_t * exp((r - 0.5*σ²)*dt + σ*√dt*Z)
    static void generate_path(double* path, double S0, double r, double sigma,
                               double T, int n_steps, const double* normals) noexcept {
        double dt     = T / static_cast<double>(n_steps);
        double drift  = (r - 0.5 * sigma * sigma) * dt;
        double vol_dt = sigma * std::sqrt(dt);
        path[0] = S0;
        for (int i = 0; i < n_steps; ++i) {
            path[i + 1] = path[i] * std::exp(drift + vol_dt * normals[i]);
        }
    }

    // Box-Muller transform: pairs of uniforms -> pairs of normals
    static void box_muller(const double* u, double* z, int n) noexcept {
        constexpr double two_pi = 6.28318530717958647692;
        for (int i = 0; i + 1 < n; i += 2) {
            double u1 = (u[i]     < 1e-15) ? 1e-15 : u[i];
            double u2 = (u[i + 1] < 1e-15) ? 1e-15 : u[i + 1];
            double rv = std::sqrt(-2.0 * std::log(u1));
            double th = two_pi * u2;
            z[i]     = rv * std::cos(th);
            z[i + 1] = rv * std::sin(th);
        }
        if (n % 2 == 1) {
            double un = (u[n - 1] < 1e-15) ? 1e-15 : u[n - 1];
            z[n - 1]  = normal_quantile(un);
        }
    }

    // Simple LCG pseudo-random for fallback (when not using Sobol)
    struct LCG {
        uint64_t state = 12345678901234567ULL;
        double next() noexcept {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            // Shift right 11 bits: 53 random bits, scale to [0,1)
            return static_cast<double>(state >> 11u) * (1.0 / static_cast<double>(1ULL << 53u));
        }
    } lcg_;
};

} // namespace options
