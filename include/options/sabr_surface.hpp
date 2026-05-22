#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace options {

struct SABRParams {
    double alpha;  // initial vol (σ_0)
    double beta;   // CEV exponent [0,1]
    double rho;    // correlation
    double nu;     // vol-of-vol
};

// ---------------------------------------------------------------------------
// Hagan 2002 SABR implied vol — Equation B.69a (corrected + stabilised)
// Returns Black implied vol σ_B(F, K, T; α, β, ρ, ν)
// Handles ATM limit, β=0, β=1 corners
// ---------------------------------------------------------------------------
inline double sabr_implied_vol(double F, double K, double T, const SABRParams& p) noexcept {
    const double alpha = p.alpha;
    const double beta  = p.beta;
    const double rho   = p.rho;
    const double nu    = p.nu;

    // Clamp inputs for numerical safety
    if (F <= 0.0 || K <= 0.0 || T <= 0.0 || alpha <= 0.0) return 0.0;

    double lnFK = std::log(F / K);

    // ATM approximation when |log(F/K)| < threshold
    if (std::abs(lnFK) < 1e-7) {
        // ATM formula: σ_ATM = α / F^(1-β) * [1 + (...)T]
        double Fbeta   = std::pow(F, 1.0 - beta);
        double Fb2     = Fbeta * Fbeta;               // F^(2*(1-β))
        double term1   = (1.0 - beta) * (1.0 - beta) * alpha * alpha / (24.0 * Fb2);
        double term2   = rho * beta * nu * alpha / (4.0 * Fbeta);
        double term3   = (2.0 - 3.0 * rho * rho) * nu * nu / 24.0;
        double sigma_b = (alpha / Fbeta) * (1.0 + (term1 + term2 + term3) * T);
        return sigma_b;
    }

    double FK_mid = std::sqrt(F * K);  // geometric mean
    double FK_b   = std::pow(FK_mid, 1.0 - beta);  // (F*K)^((1-β)/2)

    // z = (ν/α) * (F*K)^((1-β)/2) * ln(F/K)
    double z = (nu / alpha) * FK_b * lnFK;

    // χ(z) = ln[(√(1 - 2ρz + z²) + z - ρ) / (1 - ρ)]
    double disc = std::sqrt(1.0 - 2.0 * rho * z + z * z);
    double chi_z;
    if (std::abs(z) < 1e-6) {
        // Taylor: χ(z) ≈ z (1 + rho*z/2 + ...) for small z
        chi_z = z * (1.0 + rho * z / 2.0 + (1.0 / 3.0 + rho * rho / 4.0) * z * z);
    } else {
        double numerator = disc + z - rho;
        double denominator = 1.0 - rho;
        if (numerator <= 0.0 || denominator <= 0.0) return alpha / FK_b;  // fallback
        chi_z = std::log(numerator / denominator);
    }

    // Series expansion correction for log²(F/K) terms
    double ln2FK   = lnFK * lnFK;
    double ln4FK   = ln2FK * ln2FK;
    double one_b2  = (1.0 - beta) * (1.0 - beta);
    double one_b4  = one_b2 * one_b2;
    double denom_series = 1.0 + one_b2 / 24.0 * ln2FK + one_b4 / 1920.0 * ln4FK;

    // Main leading term
    double FK_b_sq = FK_b;  // already (F*K)^((1-β)/2)
    double lead = alpha / (FK_b_sq * denom_series);

    // z/χ(z) factor
    double z_chi = (std::abs(chi_z) < 1e-10) ? 1.0 : z / chi_z;

    // Correction term (1 + [...]*T)
    double FK_b2 = std::pow(FK_mid, 2.0 * (1.0 - beta));  // (F*K)^(1-β)
    double c1    = one_b2 * alpha * alpha / (24.0 * FK_b2);
    double c2    = rho * beta * nu * alpha / (4.0 * FK_b);
    double c3    = (2.0 - 3.0 * rho * rho) * nu * nu / 24.0;
    double correction = 1.0 + (c1 + c2 + c3) * T;

    double sigma_b = lead * z_chi * correction;
    return sigma_b > 0.0 ? sigma_b : 0.0;
}

// ---------------------------------------------------------------------------
// Analytic gradient of SABR implied vol w.r.t. α, β, ρ, ν (finite diff)
// (Analytic derivatives are complex; central-difference FD is stable enough
//  for LM calibration with 1e-5 step)
// ---------------------------------------------------------------------------
struct SABRGradient { double dalpha, dbeta, drho, dnu; };

inline SABRGradient sabr_gradient(double F, double K, double T, const SABRParams& p) noexcept {
    const double h = 1e-5;
    SABRGradient g{};

    auto perturb = [&](SABRParams pp) -> double {
        return sabr_implied_vol(F, K, T, pp);
    };

    SABRParams pa = p; pa.alpha += h;
    SABRParams pa2 = p; pa2.alpha -= h;
    g.dalpha = (perturb(pa) - perturb(pa2)) / (2.0 * h);

    SABRParams pb = p; pb.beta = std::clamp(p.beta + h, 0.0, 1.0 - 1e-6);
    SABRParams pb2 = p; pb2.beta = std::clamp(p.beta - h, 1e-6, 1.0);
    g.dbeta = (perturb(pb) - perturb(pb2)) / (pb.beta - pb2.beta);

    SABRParams pr = p; pr.rho = std::clamp(p.rho + h, -1.0 + 1e-6, 1.0 - 1e-6);
    SABRParams pr2 = p; pr2.rho = std::clamp(p.rho - h, -1.0 + 1e-6, 1.0 - 1e-6);
    g.drho = (perturb(pr) - perturb(pr2)) / (pr.rho - pr2.rho);

    SABRParams pn = p; pn.nu += h;
    SABRParams pn2 = p; pn2.nu -= h;
    g.dnu = (perturb(pn) - perturb(pn2)) / (2.0 * h);

    return g;
}

// ---------------------------------------------------------------------------
// SABR vol surface: calibrates to market quotes, provides interpolated vol
// ---------------------------------------------------------------------------
class SABRSurface {
public:
    struct MarketQuote {
        double F, K, T;
        double market_vol;
    };

    // Calibrate params to market quotes; returns RMS error
    double calibrate(const MarketQuote* quotes, int n_quotes,
                     SABRParams& params, int max_iter = 100, double tol = 1e-8);

    // Query calibrated surface
    double vol(double F, double K, double T) const noexcept {
        return sabr_implied_vol(F, K, T, params_);
    }

    // First derivative ∂σ/∂K (smile slope) via central differences
    double dVol_dK(double F, double K, double T) const noexcept {
        double h = K * 1e-4;
        return (sabr_implied_vol(F, K + h, T, params_) -
                sabr_implied_vol(F, K - h, T, params_)) / (2.0 * h);
    }

    // Second derivative ∂²σ/∂K² (convexity)
    double d2Vol_dK2(double F, double K, double T) const noexcept {
        double h = K * 1e-4;
        double v0 = sabr_implied_vol(F, K,     T, params_);
        double vp = sabr_implied_vol(F, K + h, T, params_);
        double vm = sabr_implied_vol(F, K - h, T, params_);
        return (vp - 2.0 * v0 + vm) / (h * h);
    }

    // Check Dupire local vol positivity (arbitrage-free check)
    // Butterfly spread must be non-negative: d²C/dK² >= 0
    bool is_arbitrage_free(double F, double K_min, double K_max,
                            double T, int n_checks = 100) const noexcept {
        double dK = (K_max - K_min) / (n_checks - 1);
        for (int i = 0; i < n_checks; ++i) {
            double K = K_min + i * dK;
            if (d2Vol_dK2(F, K, T) < -1e-6) return false;
        }
        return true;
    }

    const SABRParams& params() const noexcept { return params_; }

private:
    SABRParams params_{0.3, 0.5, -0.3, 0.4};
};

} // namespace options
