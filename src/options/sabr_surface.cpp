#include "options/sabr_surface.hpp"
#include "options/lm_optimizer.hpp"
#include <algorithm>
#include <cmath>

namespace options {

// ---------------------------------------------------------------------------
// Calibrate SABR parameters to market quotes using Levenberg-Marquardt
// Optimises all 4 params: α, β, ρ, ν
// Applies constraints: α>0, β∈[0,1], ρ∈(-1,1), ν>0
// ---------------------------------------------------------------------------
double SABRSurface::calibrate(const MarketQuote* quotes, int n_quotes,
                               SABRParams& params, int max_iter, double tol) {
    // Pack params into array
    double x[4] = { params.alpha, params.beta, params.rho, params.nu };

    // Constraint helper: project params back to feasible domain
    auto project = [](double* p) noexcept {
        p[0] = std::max(p[0], 1e-4);               // alpha > 0
        p[1] = std::clamp(p[1], 0.0, 1.0 - 1e-6); // beta in [0,1)
        p[2] = std::clamp(p[2], -1.0 + 1e-6, 1.0 - 1e-6); // rho
        p[3] = std::max(p[3], 1e-4);               // nu > 0
    };

    project(x);

    // Residual function: r_i = model_vol(i) - market_vol(i)
    auto residual = [&](const double* xx, double* f, int m) {
        SABRParams p{ xx[0], xx[1], xx[2], xx[3] };
        for (int i = 0; i < m; ++i) {
            double model = sabr_implied_vol(quotes[i].F, quotes[i].K, quotes[i].T, p);
            f[i] = model - quotes[i].market_vol;
        }
    };

    // Jacobian via central differences
    auto jacobian = [&](const double* xx, double* J, int m, int n) {
        const double h = 1e-5;
        double f_plus[512], f_minus[512];
        double xx_p[4], xx_m[4];
        for (int j = 0; j < n; ++j) {
            std::copy(xx, xx + n, xx_p);
            std::copy(xx, xx + n, xx_m);
            xx_p[j] += h;
            xx_m[j] -= h;
            project(xx_p);
            project(xx_m);
            double step = xx_p[j] - xx_m[j];
            if (step < 1e-15) step = 2.0 * h;

            SABRParams pp{ xx_p[0], xx_p[1], xx_p[2], xx_p[3] };
            SABRParams pm{ xx_m[0], xx_m[1], xx_m[2], xx_m[3] };
            for (int i = 0; i < m; ++i) {
                f_plus[i]  = sabr_implied_vol(quotes[i].F, quotes[i].K, quotes[i].T, pp);
                f_minus[i] = sabr_implied_vol(quotes[i].F, quotes[i].K, quotes[i].T, pm);
                J[i * n + j] = (f_plus[i] - f_minus[i]) / step;
            }
        }
    };

    LevenbergMarquardt lm;
    auto result = lm.optimise(x, 4, n_quotes, residual, jacobian, max_iter, tol);
    project(x);

    params_.alpha = x[0];
    params_.beta  = x[1];
    params_.rho   = x[2];
    params_.nu    = x[3];
    params = params_;

    return result.rms_error;
}

} // namespace options
