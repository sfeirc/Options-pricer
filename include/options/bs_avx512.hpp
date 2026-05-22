#pragma once
#include "bs_avx2.hpp"   // reuse exp_pd_approx, norm_cdf_avx2 types

// ---------------------------------------------------------------------------
// AVX-512 16-wide Black-Scholes
// Compile-time dispatch: if __AVX512F__ is defined (e.g. -mavx512f), the
// 16-wide path is available; otherwise falls back to two AVX2 quads.
// ---------------------------------------------------------------------------

#ifdef __AVX512F__
#include <immintrin.h>

namespace options {

// Vectorised exp for 8 doubles using AVX-512
inline __m512d exp_pd_avx512(__m512d x) noexcept {
    const __m512d inv_ln2 = _mm512_set1_pd(1.4426950408889634);
    const __m512d ln2_hi  = _mm512_set1_pd(6.93147180369123816490e-1);
    const __m512d ln2_lo  = _mm512_set1_pd(1.90821492927058770002e-10);
    const __m512d half    = _mm512_set1_pd(0.5);
    const __m512d zero    = _mm512_setzero_pd();

    __m512d kd = _mm512_floor_pd(_mm512_fmadd_pd(x, inv_ln2, half));
    __m512d r  = _mm512_fnmadd_pd(kd, ln2_hi, x);
    r          = _mm512_fnmadd_pd(kd, ln2_lo, r);

    const __m512d c0 = _mm512_set1_pd(1.0000000000000000000e+0);
    const __m512d c1 = _mm512_set1_pd(1.0000000000000000000e+0);
    const __m512d c2 = _mm512_set1_pd(4.9999999999999997158e-1);
    const __m512d c3 = _mm512_set1_pd(1.6666666666666602251e-1);
    const __m512d c4 = _mm512_set1_pd(4.1666666666667383347e-2);
    const __m512d c5 = _mm512_set1_pd(8.3333333333347225173e-3);
    const __m512d c6 = _mm512_set1_pd(1.3888888888740922013e-3);

    __m512d er = c6;
    er = _mm512_fmadd_pd(er, r, c5);
    er = _mm512_fmadd_pd(er, r, c4);
    er = _mm512_fmadd_pd(er, r, c3);
    er = _mm512_fmadd_pd(er, r, c2);
    er = _mm512_fmadd_pd(er, r, c1);
    er = _mm512_fmadd_pd(er, r, c0);

    // Scale by 2^k
    __m256i ki32   = _mm512_cvttpd_epi32(kd);
    __m512i ki64   = _mm512_cvtepi32_epi64(ki32);
    const __m512i bias = _mm512_set1_epi64(1023LL);
    __m512i exp_bits   = _mm512_slli_epi64(_mm512_add_epi64(ki64, bias), 52);
    __m512d scale      = _mm512_castsi512_pd(exp_bits);

    __m512d result = _mm512_mul_pd(er, scale);
    // Underflow: x < -745 → 0
    __mmask8 uf_mask = _mm512_cmp_pd_mask(x, _mm512_set1_pd(-745.0), _CMP_LT_OQ);
    return _mm512_mask_blend_pd(uf_mask, result, zero);
}

// Vectorised CDF for 8 doubles
inline __m512d norm_cdf_avx512(__m512d x) noexcept {
    const __m512d p_          = _mm512_set1_pd(0.2316419);
    const __m512d a1          = _mm512_set1_pd( 0.319381530);
    const __m512d a2          = _mm512_set1_pd(-0.356563782);
    const __m512d a3          = _mm512_set1_pd( 1.781477937);
    const __m512d a4          = _mm512_set1_pd(-1.821255978);
    const __m512d a5          = _mm512_set1_pd( 1.330274429);
    const __m512d one         = _mm512_set1_pd(1.0);
    const __m512d inv_sqrt2pi = _mm512_set1_pd(0.3989422804014326779);

    __m512d abs_x = _mm512_abs_pd(x);
    __m512d t     = _mm512_div_pd(one, _mm512_fmadd_pd(p_, abs_x, one));

    __m512d poly = a5;
    poly = _mm512_fmadd_pd(poly, t, a4);
    poly = _mm512_fmadd_pd(poly, t, a3);
    poly = _mm512_fmadd_pd(poly, t, a2);
    poly = _mm512_fmadd_pd(poly, t, a1);
    poly = _mm512_mul_pd(poly, t);

    __m512d neg_half_x2 = _mm512_mul_pd(_mm512_set1_pd(-0.5), _mm512_mul_pd(x, x));
    __m512d exp_val     = exp_pd_avx512(neg_half_x2);

    __m512d result = _mm512_fnmadd_pd(inv_sqrt2pi, _mm512_mul_pd(exp_val, poly), one);

    __mmask8 neg_mask = _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_LT_OQ);
    __m512d  flipped  = _mm512_sub_pd(one, result);
    return _mm512_mask_blend_pd(neg_mask, result, flipped);
}

inline __m512d norm_pdf_avx512(__m512d x) noexcept {
    const __m512d inv_sqrt2pi = _mm512_set1_pd(0.3989422804014326779);
    __m512d neg_half_x2 = _mm512_mul_pd(_mm512_set1_pd(-0.5), _mm512_mul_pd(x, x));
    return _mm512_mul_pd(inv_sqrt2pi, exp_pd_avx512(neg_half_x2));
}

// ---------------------------------------------------------------------------
// BSBatch512: 16-wide batch
// ---------------------------------------------------------------------------
struct BSBatch512 {
    alignas(64) double S[16];
    alignas(64) double K[16];
    alignas(64) double T[16];
    alignas(64) double r[16];
    alignas(64) double sigma[16];
    alignas(64) double is_call[16];

    alignas(64) double price[16];
    alignas(64) double delta[16];
    alignas(64) double gamma[16];
    alignas(64) double vega[16];
    alignas(64) double theta[16];
    alignas(64) double rho[16];
};

// Internal: price 8 options with AVX-512 (one __m512d lane)
inline void bs_avx512_octet(
    __m512d vS, __m512d vK, __m512d vT, __m512d vr,
    __m512d vsigma, __m512d vis_call,
    double* price_out, double* delta_out, double* gamma_out,
    double* vega_out,  double* theta_out, double* rho_out) noexcept
{
    const __m512d zero  = _mm512_setzero_pd();
    const __m512d one   = _mm512_set1_pd(1.0);
    const __m512d half  = _mm512_set1_pd(0.5);
    const __m512d two   = _mm512_set1_pd(2.0);

    __m512d sqrt_T = _mm512_sqrt_pd(vT);

    // Scalar log per lane
    alignas(64) double s_arr[8], k_arr[8], logSK[8];
    _mm512_store_pd(s_arr, vS);
    _mm512_store_pd(k_arr, vK);
    for (int i = 0; i < 8; ++i) logSK[i] = std::log(s_arr[i] / k_arr[i]);
    __m512d vlogSK = _mm512_load_pd(logSK);

    __m512d sigma2  = _mm512_mul_pd(vsigma, vsigma);
    __m512d drift   = _mm512_fmadd_pd(half, sigma2, vr);
    __m512d d1_num  = _mm512_fmadd_pd(drift, vT, vlogSK);
    __m512d sig_sqT = _mm512_mul_pd(vsigma, sqrt_T);
    __m512d d1      = _mm512_div_pd(d1_num, sig_sqT);
    __m512d d2      = _mm512_sub_pd(d1, sig_sqT);

    __m512d Nd1   = norm_cdf_avx512(d1);
    __m512d Nd2   = norm_cdf_avx512(d2);
    __m512d nd1   = norm_pdf_avx512(d1);
    __m512d Nm_d1 = _mm512_sub_pd(one, Nd1);
    __m512d Nm_d2 = _mm512_sub_pd(one, Nd2);

    __m512d neg_rT = _mm512_mul_pd(_mm512_sub_pd(zero, vr), vT);
    __m512d disc   = exp_pd_avx512(neg_rT);
    __m512d Kdisc  = _mm512_mul_pd(vK, disc);

    __m512d call_price = _mm512_fmsub_pd(vS, Nd1, _mm512_mul_pd(Kdisc, Nd2));
    __m512d call_delta = Nd1;
    __m512d nd1_sig_2sqT = _mm512_div_pd(
        _mm512_mul_pd(_mm512_mul_pd(vS, nd1), vsigma),
        _mm512_mul_pd(two, sqrt_T));
    __m512d call_theta = _mm512_sub_pd(
        _mm512_sub_pd(zero, nd1_sig_2sqT),
        _mm512_mul_pd(vr, _mm512_mul_pd(Kdisc, Nd2)));
    __m512d call_rho = _mm512_mul_pd(vT, _mm512_mul_pd(Kdisc, Nd2));

    __m512d put_price = _mm512_fmsub_pd(Kdisc, Nm_d2, _mm512_mul_pd(vS, Nm_d1));
    __m512d put_delta = _mm512_sub_pd(Nd1, one);
    __m512d put_theta = _mm512_add_pd(
        _mm512_sub_pd(zero, nd1_sig_2sqT),
        _mm512_mul_pd(vr, _mm512_mul_pd(Kdisc, Nm_d2)));
    __m512d put_rho   = _mm512_sub_pd(zero,
        _mm512_mul_pd(vT, _mm512_mul_pd(Kdisc, Nm_d2)));

    __mmask8 call_mask = _mm512_cmp_pd_mask(vis_call, half, _CMP_GT_OQ);
    __m512d price_v = _mm512_mask_blend_pd(call_mask, put_price,  call_price);
    __m512d delta_v = _mm512_mask_blend_pd(call_mask, put_delta,  call_delta);
    __m512d theta_v = _mm512_mask_blend_pd(call_mask, put_theta,  call_theta);
    __m512d rho_v   = _mm512_mask_blend_pd(call_mask, put_rho,    call_rho);

    __m512d gamma_v = _mm512_div_pd(nd1, _mm512_mul_pd(vS, sig_sqT));
    __m512d vega_v  = _mm512_mul_pd(vS, _mm512_mul_pd(nd1, sqrt_T));

    _mm512_store_pd(price_out, price_v);
    _mm512_store_pd(delta_out, delta_v);
    _mm512_store_pd(gamma_out, gamma_v);
    _mm512_store_pd(vega_out,  vega_v);
    _mm512_store_pd(theta_out, theta_v);
    _mm512_store_pd(rho_out,   rho_v);
}

// Price 16 options in one pass
inline void bs_avx512_batch(BSBatch512& batch) noexcept {
    // Lane 0-7
    bs_avx512_octet(
        _mm512_load_pd(batch.S),
        _mm512_load_pd(batch.K),
        _mm512_load_pd(batch.T),
        _mm512_load_pd(batch.r),
        _mm512_load_pd(batch.sigma),
        _mm512_load_pd(batch.is_call),
        batch.price, batch.delta, batch.gamma,
        batch.vega,  batch.theta, batch.rho);
    // Lane 8-15
    bs_avx512_octet(
        _mm512_load_pd(batch.S      + 8),
        _mm512_load_pd(batch.K      + 8),
        _mm512_load_pd(batch.T      + 8),
        _mm512_load_pd(batch.r      + 8),
        _mm512_load_pd(batch.sigma  + 8),
        _mm512_load_pd(batch.is_call + 8),
        batch.price + 8, batch.delta + 8, batch.gamma + 8,
        batch.vega  + 8, batch.theta + 8, batch.rho   + 8);
}

} // namespace options

#else  // fallback to double-AVX2

namespace options {

// BSBatch512 with AVX2 fallback implementation
struct BSBatch512 {
    alignas(64) double S[16];
    alignas(64) double K[16];
    alignas(64) double T[16];
    alignas(64) double r[16];
    alignas(64) double sigma[16];
    alignas(64) double is_call[16];

    alignas(64) double price[16];
    alignas(64) double delta[16];
    alignas(64) double gamma[16];
    alignas(64) double vega[16];
    alignas(64) double theta[16];
    alignas(64) double rho[16];
};

inline void bs_avx512_batch(BSBatch512& batch) noexcept {
    // Run two BSBatch (8-wide each) back-to-back
    BSBatch b0, b1;
    for (int i = 0; i < 8; ++i) {
        b0.S[i] = batch.S[i];       b1.S[i] = batch.S[i+8];
        b0.K[i] = batch.K[i];       b1.K[i] = batch.K[i+8];
        b0.T[i] = batch.T[i];       b1.T[i] = batch.T[i+8];
        b0.r[i] = batch.r[i];       b1.r[i] = batch.r[i+8];
        b0.sigma[i]   = batch.sigma[i];    b1.sigma[i]   = batch.sigma[i+8];
        b0.is_call[i] = batch.is_call[i];  b1.is_call[i] = batch.is_call[i+8];
    }
    bs_avx2_batch(b0);
    bs_avx2_batch(b1);
    for (int i = 0; i < 8; ++i) {
        batch.price[i]  = b0.price[i];  batch.price[i+8]  = b1.price[i];
        batch.delta[i]  = b0.delta[i];  batch.delta[i+8]  = b1.delta[i];
        batch.gamma[i]  = b0.gamma[i];  batch.gamma[i+8]  = b1.gamma[i];
        batch.vega[i]   = b0.vega[i];   batch.vega[i+8]   = b1.vega[i];
        batch.theta[i]  = b0.theta[i];  batch.theta[i+8]  = b1.theta[i];
        batch.rho[i]    = b0.rho[i];    batch.rho[i+8]    = b1.rho[i];
    }
}

} // namespace options

#endif // __AVX512F__
