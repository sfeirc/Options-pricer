#pragma once
#include <cmath>

namespace options {

struct BSResult {
    double price, delta, gamma, vega, theta, rho, vanna, volga;
};

// Fast cumulative normal via minimax polynomial (Abramowitz & Stegun 26.2.17)
// Max error 7.5e-8
inline double norm_cdf(double x) noexcept {
    constexpr double a1 =  0.319381530;
    constexpr double a2 = -0.356563782;
    constexpr double a3 =  1.781477937;
    constexpr double a4 = -1.821255978;
    constexpr double a5 =  1.330274429;
    constexpr double p  =  0.2316419;
    double t = 1.0 / (1.0 + p * std::abs(x));
    double poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    double result = 1.0 - (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x) * poly;
    return (x >= 0.0) ? result : 1.0 - result;
}

inline double norm_pdf(double x) noexcept {
    return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
}

inline BSResult bs_price(double S, double K, double T, double r, double sigma, bool is_call) noexcept {
    BSResult res{};
    if (T <= 0.0 || sigma <= 0.0) {
        res.price = is_call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
        return res;
    }
    double sqrt_T = std::sqrt(T);
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
    double d2 = d1 - sigma * sqrt_T;
    double Nd1 = norm_cdf(d1), Nd2 = norm_cdf(d2);
    double nd1 = norm_pdf(d1);
    double disc = std::exp(-r * T);

    if (is_call) {
        res.price  = S * Nd1 - K * disc * Nd2;
        res.delta  = Nd1;
        res.theta  = -(S * nd1 * sigma) / (2.0 * sqrt_T) - r * K * disc * Nd2;
        res.rho    = K * T * disc * Nd2;
    } else {
        double Nm_d1 = 1.0 - Nd1, Nm_d2 = 1.0 - Nd2;
        res.price  = K * disc * Nm_d2 - S * Nm_d1;
        res.delta  = Nd1 - 1.0;
        res.theta  = -(S * nd1 * sigma) / (2.0 * sqrt_T) + r * K * disc * Nm_d2;
        res.rho    = -K * T * disc * Nm_d2;
    }
    res.gamma = nd1 / (S * sigma * sqrt_T);
    res.vega  = S * nd1 * sqrt_T;
    res.vanna = -nd1 * d2 / sigma;
    res.volga = res.vega * d1 * d2 / sigma;
    return res;
}

} // namespace options
