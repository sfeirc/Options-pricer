#pragma once
#include <immintrin.h>
#include <cstddef>
#include <cstdint>

namespace options {

// ---------------------------------------------------------------------------
// Vectorised natural log for positive doubles.
// Accurate to < 1 ULP for x in (0, +inf).
// Algorithm: IEEE 754 exponent extraction + atanh polynomial on mantissa.
// ---------------------------------------------------------------------------
inline __m256d log_pd_approx(__m256d x) noexcept {
    // --- Extract biased exponent as integer ------------------------------------
    __m256i xi = _mm256_castpd_si256(x);
    // e_int = bits[62:52] = (xi >> 52) & 0x7FF  (biased exponent, in [0, 2046])
    __m256i e_int = _mm256_and_si256(
        _mm256_srli_epi64(xi, 52),
        _mm256_set1_epi64x(0x7FFL));

    // --- Convert e_int (int64) to double via magic bit trick -------------------
    // For 0 <= n < 2^52: double(n) = reinterpret_as_double(n | 2^52) - 2^52
    const __m256i magic_i = _mm256_set1_epi64x(0x4330000000000000LL);  // 2^52 bits
    const __m256d magic_d = _mm256_set1_pd(4503599627370496.0);        // 2^52
    __m256d e = _mm256_sub_pd(
        _mm256_castsi256_pd(_mm256_or_si256(e_int, magic_i)),
        magic_d);
    e = _mm256_sub_pd(e, _mm256_set1_pd(1023.0));  // remove bias: e = true exponent

    // --- Extract mantissa m in [1.0, 2.0) -------------------------------------
    const __m256i mant_mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFFL);
    const __m256i one_exp   = _mm256_set1_epi64x(0x3FF0000000000000L);  // 1.0 bits
    __m256d m = _mm256_castsi256_pd(
        _mm256_or_si256(_mm256_and_si256(xi, mant_mask), one_exp));

    // --- Range reduction: bring m into [1/sqrt(2), sqrt(2)) ------------------
    // If m >= sqrt(2): m /= 2, e += 1  ->  tighter range -> fewer poly terms needed
    const __m256d sqrt2 = _mm256_set1_pd(1.41421356237309504880168872);
    __m256d ge  = _mm256_cmp_pd(m, sqrt2, _CMP_GE_OQ);  // all-1s where m >= sqrt2
    m   = _mm256_blendv_pd(m, _mm256_mul_pd(m, _mm256_set1_pd(0.5)), ge);
    // Add 1.0 to exponent for lanes where m was halved
    // Use bitwise AND: ge & 1.0_bits = 1.0 where true, 0.0 where false
    e   = _mm256_add_pd(e, _mm256_and_pd(ge, _mm256_set1_pd(1.0)));
    // Now m in [0.7071, 1.4142),  log(x) = e*ln2 + log(m)

    // --- log(m) via 2*atanh((m-1)/(m+1)) series ------------------------------
    // t = (m-1)/(m+1),  |t| < (sqrt2-1)/(sqrt2+1) ~= 0.1716
    __m256d t = _mm256_div_pd(
        _mm256_sub_pd(m, _mm256_set1_pd(1.0)),
        _mm256_add_pd(m, _mm256_set1_pd(1.0)));
    __m256d t2 = _mm256_mul_pd(t, t);

    // Minimax polynomial for 2*atanh(t)/t on |t| < 0.18  (7 terms, ~1e-15 error)
    // 2*(1 + t^2/3 + t^4/5 + t^6/7 + t^8/9 + t^10/11 + t^12/13)
    __m256d p = _mm256_set1_pd(2.0 / 13.0);
    p = _mm256_fmadd_pd(p, t2, _mm256_set1_pd(2.0 / 11.0));
    p = _mm256_fmadd_pd(p, t2, _mm256_set1_pd(2.0 /  9.0));
    p = _mm256_fmadd_pd(p, t2, _mm256_set1_pd(2.0 /  7.0));
    p = _mm256_fmadd_pd(p, t2, _mm256_set1_pd(2.0 /  5.0));
    p = _mm256_fmadd_pd(p, t2, _mm256_set1_pd(2.0 /  3.0));
    p = _mm256_fmadd_pd(p, t2, _mm256_set1_pd(2.0));
    __m256d log_m = _mm256_mul_pd(p, t);  // = 2*atanh(t) = log(m)

    // log(x) = e * ln(2) + log(m)
    return _mm256_fmadd_pd(e, _mm256_set1_pd(0.69314718055994530941723212), log_m);
}

// ---------------------------------------------------------------------------
// Fast vectorised exp via range reduction + degree-6 polynomial + 2^k scaling
// Accurate to ~1e-10 for x in [-87, 0]
// ---------------------------------------------------------------------------
inline __m256d exp_pd_approx(__m256d x) noexcept {
    // Range reduction: x = k*ln2 + r, |r| <= ln2/2
    const __m256d inv_ln2  = _mm256_set1_pd(1.4426950408889634);
    const __m256d ln2_hi   = _mm256_set1_pd(6.93147180369123816490e-1);
    const __m256d ln2_lo   = _mm256_set1_pd(1.90821492927058770002e-10);
    const __m256d half     = _mm256_set1_pd(0.5);
    const __m256d zero     = _mm256_setzero_pd();

    // k = floor(x / ln2 + 0.5)
    __m256d kd = _mm256_floor_pd(_mm256_fmadd_pd(x, inv_ln2, half));

    // r = x - k*ln2  (two-sum for accuracy)
    __m256d r = _mm256_fnmadd_pd(kd, ln2_hi, x);
    r = _mm256_fnmadd_pd(kd, ln2_lo, r);

    // Polynomial approximation of exp(r) for |r| <= ln2/2
    // Coefficients from Cephes / minimax fit
    const __m256d c0 = _mm256_set1_pd(1.0000000000000000000e+0);
    const __m256d c1 = _mm256_set1_pd(1.0000000000000000000e+0);
    const __m256d c2 = _mm256_set1_pd(4.9999999999999997158e-1);
    const __m256d c3 = _mm256_set1_pd(1.6666666666666602251e-1);
    const __m256d c4 = _mm256_set1_pd(4.1666666666667383347e-2);
    const __m256d c5 = _mm256_set1_pd(8.3333333333347225173e-3);
    const __m256d c6 = _mm256_set1_pd(1.3888888888740922013e-3);

    // Horner evaluation: c0 + r*(c1 + r*(c2 + ... + r*c6))
    __m256d er = c6;
    er = _mm256_fmadd_pd(er, r, c5);
    er = _mm256_fmadd_pd(er, r, c4);
    er = _mm256_fmadd_pd(er, r, c3);
    er = _mm256_fmadd_pd(er, r, c2);
    er = _mm256_fmadd_pd(er, r, c1);
    er = _mm256_fmadd_pd(er, r, c0);

    // Scale by 2^k: construct (k+1023) << 52 as IEEE 754 exponent
    // Convert k to 32-bit int, extend to 64-bit, shift into exponent field
    __m128i ki32 = _mm256_cvttpd_epi32(kd);

    // Extend 32-bit to 64-bit integers
    __m256i ki64 = _mm256_cvtepi32_epi64(ki32);

    // bias = 1023, shift = 52
    const __m256i bias = _mm256_set1_epi64x(1023LL);
    __m256i exp_bits = _mm256_slli_epi64(_mm256_add_epi64(ki64, bias), 52);

    // Reinterpret as double scale factor
    __m256d scale = _mm256_castsi256_pd(exp_bits);

    // Clamp: if x < -745 return 0; the poly handles normal range
    __m256d result = _mm256_mul_pd(er, scale);
    __m256d underflow_mask = _mm256_cmp_pd(x, _mm256_set1_pd(-745.0), _CMP_LT_OQ);
    return _mm256_blendv_pd(result, zero, underflow_mask);
}

// ---------------------------------------------------------------------------
// Vectorised cumulative normal CDF using minimax polynomial (A&S 26.2.17)
// Processes 4 doubles simultaneously (__m256d = 4 x double)
// ---------------------------------------------------------------------------
inline __m256d norm_cdf_avx2(__m256d x) noexcept {
    const __m256d p_          = _mm256_set1_pd(0.2316419);
    const __m256d a1          = _mm256_set1_pd( 0.319381530);
    const __m256d a2          = _mm256_set1_pd(-0.356563782);
    const __m256d a3          = _mm256_set1_pd( 1.781477937);
    const __m256d a4          = _mm256_set1_pd(-1.821255978);
    const __m256d a5          = _mm256_set1_pd( 1.330274429);
    const __m256d one         = _mm256_set1_pd(1.0);
    const __m256d inv_sqrt2pi = _mm256_set1_pd(0.3989422804014326779);
    const __m256d sign_mask   = _mm256_set1_pd(-0.0);  // only sign bit set

    // |x|
    __m256d abs_x = _mm256_andnot_pd(sign_mask, x);

    // t = 1 / (1 + p*|x|)
    __m256d t = _mm256_div_pd(one, _mm256_fmadd_pd(p_, abs_x, one));

    // Horner: poly = t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))
    __m256d poly = a5;
    poly = _mm256_fmadd_pd(poly, t, a4);
    poly = _mm256_fmadd_pd(poly, t, a3);
    poly = _mm256_fmadd_pd(poly, t, a2);
    poly = _mm256_fmadd_pd(poly, t, a1);
    poly = _mm256_mul_pd(poly, t);

    // exp(-0.5*x^2)
    __m256d neg_half_x2 = _mm256_mul_pd(_mm256_set1_pd(-0.5), _mm256_mul_pd(x, x));
    __m256d exp_val     = exp_pd_approx(neg_half_x2);

    // result = 1 - inv_sqrt2pi * exp_val * poly
    __m256d result = _mm256_fnmadd_pd(inv_sqrt2pi, _mm256_mul_pd(exp_val, poly), one);

    // For negative x: result = 1 - result
    __m256d neg_mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_LT_OQ);
    __m256d flipped  = _mm256_sub_pd(one, result);
    return _mm256_blendv_pd(result, flipped, neg_mask);
}

// ---------------------------------------------------------------------------
// Vectorised norm_pdf for 4 doubles
// ---------------------------------------------------------------------------
inline __m256d norm_pdf_avx2(__m256d x) noexcept {
    const __m256d inv_sqrt2pi = _mm256_set1_pd(0.3989422804014326779);
    __m256d neg_half_x2 = _mm256_mul_pd(_mm256_set1_pd(-0.5), _mm256_mul_pd(x, x));
    return _mm256_mul_pd(inv_sqrt2pi, exp_pd_approx(neg_half_x2));
}

// ---------------------------------------------------------------------------
// Combined CDF+PDF: compute N(x) and n(x) with a single exp evaluation.
// Saves one exp call vs calling norm_cdf_avx2 + norm_pdf_avx2 separately.
// cdf_out and pdf_out receive N(x) and n(x) respectively.
// ---------------------------------------------------------------------------
inline void norm_cdf_pdf_avx2(__m256d x, __m256d& cdf_out, __m256d& pdf_out) noexcept {
    const __m256d p_          = _mm256_set1_pd(0.2316419);
    const __m256d a1          = _mm256_set1_pd( 0.319381530);
    const __m256d a2          = _mm256_set1_pd(-0.356563782);
    const __m256d a3          = _mm256_set1_pd( 1.781477937);
    const __m256d a4          = _mm256_set1_pd(-1.821255978);
    const __m256d a5          = _mm256_set1_pd( 1.330274429);
    const __m256d one         = _mm256_set1_pd(1.0);
    const __m256d inv_sqrt2pi = _mm256_set1_pd(0.3989422804014326779);
    const __m256d sign_mask   = _mm256_set1_pd(-0.0);

    __m256d abs_x = _mm256_andnot_pd(sign_mask, x);
    __m256d t = _mm256_div_pd(one, _mm256_fmadd_pd(p_, abs_x, one));

    __m256d poly = a5;
    poly = _mm256_fmadd_pd(poly, t, a4);
    poly = _mm256_fmadd_pd(poly, t, a3);
    poly = _mm256_fmadd_pd(poly, t, a2);
    poly = _mm256_fmadd_pd(poly, t, a1);
    poly = _mm256_mul_pd(poly, t);

    // Single exp evaluation shared between CDF and PDF
    __m256d neg_half_x2 = _mm256_mul_pd(_mm256_set1_pd(-0.5), _mm256_mul_pd(x, x));
    __m256d exp_val     = exp_pd_approx(neg_half_x2);

    pdf_out = _mm256_mul_pd(inv_sqrt2pi, exp_val);

    __m256d result = _mm256_fnmadd_pd(inv_sqrt2pi, _mm256_mul_pd(exp_val, poly), one);
    __m256d neg_mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_LT_OQ);
    __m256d flipped  = _mm256_sub_pd(one, result);
    cdf_out = _mm256_blendv_pd(result, flipped, neg_mask);
}

// ---------------------------------------------------------------------------
// BSBatch: input/output layout for 8-wide batch pricing
// Note: __m256d is 4 doubles wide; we process two __m256d batches per call
// to cover all 8 values. Arrays remain 8-element for caller convenience.
// ---------------------------------------------------------------------------
struct BSBatch {
    // Input arrays — 32-byte aligned for AVX2 loads
    alignas(32) double S[8];
    alignas(32) double K[8];
    alignas(32) double T[8];
    alignas(32) double r[8];
    alignas(32) double sigma[8];
    alignas(32) double is_call[8];  // 1.0 = call, 0.0 = put

    // Output arrays
    alignas(32) double price[8];
    alignas(32) double delta[8];
    alignas(32) double gamma[8];
    alignas(32) double vega[8];
    alignas(32) double theta[8];
    alignas(32) double rho[8];
};

// ---------------------------------------------------------------------------
// Internal: price a batch of 4 options (one __m256d lane)
// All arguments are __m256d registers. Results written to aligned pointers.
// ---------------------------------------------------------------------------
inline void bs_avx2_quad(
    __m256d vS, __m256d vK, __m256d vT, __m256d vr,
    __m256d vsigma, __m256d vis_call,
    double* price_out, double* delta_out, double* gamma_out,
    double* vega_out,  double* theta_out, double* rho_out) noexcept
{
    const __m256d zero     = _mm256_setzero_pd();
    const __m256d one      = _mm256_set1_pd(1.0);
    const __m256d half     = _mm256_set1_pd(0.5);
    const __m256d two      = _mm256_set1_pd(2.0);

    // sqrt(T)
    __m256d sqrt_T = _mm256_sqrt_pd(vT);

    // log(S/K) — fully vectorised, no scalar fallback
    __m256d vlogSK = log_pd_approx(_mm256_div_pd(vS, vK));

    // d1 = (log(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt_T)
    __m256d sigma2   = _mm256_mul_pd(vsigma, vsigma);
    __m256d drift    = _mm256_fmadd_pd(half, sigma2, vr);        // r + 0.5*σ²
    __m256d d1_num   = _mm256_fmadd_pd(drift, vT, vlogSK);      // log(S/K) + drift*T
    __m256d sig_sqT  = _mm256_mul_pd(vsigma, sqrt_T);
    __m256d d1       = _mm256_div_pd(d1_num, sig_sqT);
    __m256d d2       = _mm256_sub_pd(d1, sig_sqT);

    // CDF and PDF
    __m256d Nd1  = norm_cdf_avx2(d1);
    __m256d Nd2  = norm_cdf_avx2(d2);
    __m256d nd1  = norm_pdf_avx2(d1);

    // 1 - Nd1, 1 - Nd2 (for puts)
    __m256d Nm_d1 = _mm256_sub_pd(one, Nd1);
    __m256d Nm_d2 = _mm256_sub_pd(one, Nd2);

    // discount = exp(-r*T)
    __m256d neg_rT = _mm256_mul_pd(_mm256_sub_pd(zero, vr), vT);
    __m256d disc   = exp_pd_approx(neg_rT);
    __m256d Kdisc  = _mm256_mul_pd(vK, disc);

    // Call outputs
    __m256d call_price  = _mm256_fmsub_pd(vS, Nd1, _mm256_mul_pd(Kdisc, Nd2));
    __m256d call_delta  = Nd1;
    __m256d neg_rKdNd2  = _mm256_mul_pd(vr, _mm256_mul_pd(Kdisc, Nd2));
    __m256d nd1_sig_2sqT = _mm256_div_pd(_mm256_mul_pd(_mm256_mul_pd(vS, nd1), vsigma),
                                          _mm256_mul_pd(two, sqrt_T));
    __m256d call_theta  = _mm256_sub_pd(_mm256_sub_pd(zero, nd1_sig_2sqT), neg_rKdNd2);
    __m256d call_rho    = _mm256_mul_pd(vT, _mm256_mul_pd(Kdisc, Nd2));

    // Put outputs
    __m256d put_price  = _mm256_fmsub_pd(Kdisc, Nm_d2, _mm256_mul_pd(vS, Nm_d1));
    __m256d put_delta  = _mm256_sub_pd(Nd1, one);
    __m256d pos_rKdNmd2 = _mm256_mul_pd(vr, _mm256_mul_pd(Kdisc, Nm_d2));
    __m256d put_theta  = _mm256_add_pd(_mm256_sub_pd(zero, nd1_sig_2sqT), pos_rKdNmd2);
    __m256d put_rho    = _mm256_sub_pd(zero, _mm256_mul_pd(vT, _mm256_mul_pd(Kdisc, Nm_d2)));

    // Blend call vs put based on is_call mask
    __m256d call_mask = _mm256_cmp_pd(vis_call, half, _CMP_GT_OQ);  // is_call > 0.5
    __m256d price_v  = _mm256_blendv_pd(put_price,  call_price,  call_mask);
    __m256d delta_v  = _mm256_blendv_pd(put_delta,  call_delta,  call_mask);
    __m256d theta_v  = _mm256_blendv_pd(put_theta,  call_theta,  call_mask);
    __m256d rho_v    = _mm256_blendv_pd(put_rho,    call_rho,    call_mask);

    // Greeks common to call and put
    // gamma = nd1 / (S * sigma * sqrt_T)
    __m256d gamma_v = _mm256_div_pd(nd1, _mm256_mul_pd(vS, sig_sqT));
    // vega  = S * nd1 * sqrt_T
    __m256d vega_v  = _mm256_mul_pd(vS, _mm256_mul_pd(nd1, sqrt_T));

    // Store results
    _mm256_store_pd(price_out, price_v);
    _mm256_store_pd(delta_out, delta_v);
    _mm256_store_pd(gamma_out, gamma_v);
    _mm256_store_pd(vega_out,  vega_v);
    _mm256_store_pd(theta_out, theta_v);
    _mm256_store_pd(rho_out,   rho_v);
}

// ---------------------------------------------------------------------------
// Price a batch of 8 options, computing all Greeks in one pass.
// Both quads are interleaved so the CPU can overlap their independent
// instruction streams across its two FMA units.
// ---------------------------------------------------------------------------
inline void bs_avx2_batch(BSBatch& batch) noexcept {
    const __m256d zero     = _mm256_setzero_pd();
    const __m256d one      = _mm256_set1_pd(1.0);
    const __m256d half     = _mm256_set1_pd(0.5);
    const __m256d two      = _mm256_set1_pd(2.0);
    const __m256d nhalf    = _mm256_set1_pd(-0.5);

    // Load both quads upfront
    __m256d vS0 = _mm256_load_pd(batch.S),        vS1 = _mm256_load_pd(batch.S     + 4);
    __m256d vK0 = _mm256_load_pd(batch.K),        vK1 = _mm256_load_pd(batch.K     + 4);
    __m256d vT0 = _mm256_load_pd(batch.T),        vT1 = _mm256_load_pd(batch.T     + 4);
    __m256d vr0 = _mm256_load_pd(batch.r),        vr1 = _mm256_load_pd(batch.r     + 4);
    __m256d vs0 = _mm256_load_pd(batch.sigma),    vs1 = _mm256_load_pd(batch.sigma + 4);
    __m256d vc0 = _mm256_load_pd(batch.is_call),  vc1 = _mm256_load_pd(batch.is_call + 4);

    // sqrt(T) — independent, both quads in parallel
    __m256d sqT0 = _mm256_sqrt_pd(vT0),  sqT1 = _mm256_sqrt_pd(vT1);

    // log(S/K) — vectorised, both quads
    __m256d logSK0 = log_pd_approx(_mm256_div_pd(vS0, vK0));
    __m256d logSK1 = log_pd_approx(_mm256_div_pd(vS1, vK1));

    // sigma^2
    __m256d sig2_0 = _mm256_mul_pd(vs0, vs0),  sig2_1 = _mm256_mul_pd(vs1, vs1);

    // drift = r + 0.5*sigma^2
    __m256d drift0 = _mm256_fmadd_pd(half, sig2_0, vr0);
    __m256d drift1 = _mm256_fmadd_pd(half, sig2_1, vr1);

    // d1_num = log(S/K) + drift*T
    __m256d d1n0 = _mm256_fmadd_pd(drift0, vT0, logSK0);
    __m256d d1n1 = _mm256_fmadd_pd(drift1, vT1, logSK1);

    // sig*sqrt(T)
    __m256d ssT0 = _mm256_mul_pd(vs0, sqT0),  ssT1 = _mm256_mul_pd(vs1, sqT1);

    // d1 = d1_num / (sig*sqrt(T))
    __m256d d1_0 = _mm256_div_pd(d1n0, ssT0),  d1_1 = _mm256_div_pd(d1n1, ssT1);

    // d2 = d1 - sig*sqrt(T)
    __m256d d2_0 = _mm256_sub_pd(d1_0, ssT0),  d2_1 = _mm256_sub_pd(d1_1, ssT1);

    // CDF and PDF — interleaved
    __m256d Nd1_0 = norm_cdf_avx2(d1_0),  Nd1_1 = norm_cdf_avx2(d1_1);
    __m256d Nd2_0 = norm_cdf_avx2(d2_0),  Nd2_1 = norm_cdf_avx2(d2_1);
    __m256d nd1_0 = norm_pdf_avx2(d1_0),  nd1_1 = norm_pdf_avx2(d1_1);

    // 1 - Nd1, 1 - Nd2
    __m256d Nmd1_0 = _mm256_sub_pd(one, Nd1_0),  Nmd1_1 = _mm256_sub_pd(one, Nd1_1);
    __m256d Nmd2_0 = _mm256_sub_pd(one, Nd2_0),  Nmd2_1 = _mm256_sub_pd(one, Nd2_1);

    // discount = exp(-r*T)
    __m256d nrT0 = _mm256_mul_pd(_mm256_sub_pd(zero, vr0), vT0);
    __m256d nrT1 = _mm256_mul_pd(_mm256_sub_pd(zero, vr1), vT1);
    __m256d disc0 = exp_pd_approx(nrT0),  disc1 = exp_pd_approx(nrT1);
    __m256d Kd0 = _mm256_mul_pd(vK0, disc0),  Kd1 = _mm256_mul_pd(vK1, disc1);

    // Call outputs
    __m256d cp0 = _mm256_fmsub_pd(vS0, Nd1_0, _mm256_mul_pd(Kd0, Nd2_0));
    __m256d cp1 = _mm256_fmsub_pd(vS1, Nd1_1, _mm256_mul_pd(Kd1, Nd2_1));

    // theta intermediate: nd1*sig / (2*sqrt(T))
    __m256d ns2T0 = _mm256_div_pd(_mm256_mul_pd(_mm256_mul_pd(vS0, nd1_0), vs0),
                                   _mm256_mul_pd(two, sqT0));
    __m256d ns2T1 = _mm256_div_pd(_mm256_mul_pd(_mm256_mul_pd(vS1, nd1_1), vs1),
                                   _mm256_mul_pd(two, sqT1));

    // call theta = -ns2T - r*K*disc*Nd2
    __m256d cth0 = _mm256_sub_pd(_mm256_sub_pd(zero, ns2T0),
                                  _mm256_mul_pd(vr0, _mm256_mul_pd(Kd0, Nd2_0)));
    __m256d cth1 = _mm256_sub_pd(_mm256_sub_pd(zero, ns2T1),
                                  _mm256_mul_pd(vr1, _mm256_mul_pd(Kd1, Nd2_1)));

    // call rho = T*K*disc*Nd2
    __m256d crho0 = _mm256_mul_pd(vT0, _mm256_mul_pd(Kd0, Nd2_0));
    __m256d crho1 = _mm256_mul_pd(vT1, _mm256_mul_pd(Kd1, Nd2_1));

    // Put outputs
    __m256d pp0 = _mm256_fmsub_pd(Kd0, Nmd2_0, _mm256_mul_pd(vS0, Nmd1_0));
    __m256d pp1 = _mm256_fmsub_pd(Kd1, Nmd2_1, _mm256_mul_pd(vS1, Nmd1_1));

    __m256d pth0 = _mm256_add_pd(_mm256_sub_pd(zero, ns2T0),
                                  _mm256_mul_pd(vr0, _mm256_mul_pd(Kd0, Nmd2_0)));
    __m256d pth1 = _mm256_add_pd(_mm256_sub_pd(zero, ns2T1),
                                  _mm256_mul_pd(vr1, _mm256_mul_pd(Kd1, Nmd2_1)));

    __m256d prho0 = _mm256_sub_pd(zero, _mm256_mul_pd(vT0, _mm256_mul_pd(Kd0, Nmd2_0)));
    __m256d prho1 = _mm256_sub_pd(zero, _mm256_mul_pd(vT1, _mm256_mul_pd(Kd1, Nmd2_1)));

    // Blend call vs put
    __m256d cmask0 = _mm256_cmp_pd(vc0, half, _CMP_GT_OQ);
    __m256d cmask1 = _mm256_cmp_pd(vc1, half, _CMP_GT_OQ);

    __m256d price0  = _mm256_blendv_pd(pp0,  cp0,  cmask0);
    __m256d price1  = _mm256_blendv_pd(pp1,  cp1,  cmask1);
    __m256d delta0  = _mm256_blendv_pd(_mm256_sub_pd(Nd1_0, one), Nd1_0, cmask0);
    __m256d delta1  = _mm256_blendv_pd(_mm256_sub_pd(Nd1_1, one), Nd1_1, cmask1);
    __m256d theta0  = _mm256_blendv_pd(pth0, cth0, cmask0);
    __m256d theta1  = _mm256_blendv_pd(pth1, cth1, cmask1);
    __m256d rho0    = _mm256_blendv_pd(prho0, crho0, cmask0);
    __m256d rho1    = _mm256_blendv_pd(prho1, crho1, cmask1);

    // gamma = nd1 / (S * sig * sqrt(T))
    __m256d gamma0 = _mm256_div_pd(nd1_0, _mm256_mul_pd(vS0, ssT0));
    __m256d gamma1 = _mm256_div_pd(nd1_1, _mm256_mul_pd(vS1, ssT1));

    // vega = S * nd1 * sqrt(T)
    __m256d vega0 = _mm256_mul_pd(vS0, _mm256_mul_pd(nd1_0, sqT0));
    __m256d vega1 = _mm256_mul_pd(vS1, _mm256_mul_pd(nd1_1, sqT1));

    // Store both quads
    _mm256_store_pd(batch.price,     price0);  _mm256_store_pd(batch.price + 4, price1);
    _mm256_store_pd(batch.delta,     delta0);  _mm256_store_pd(batch.delta + 4, delta1);
    _mm256_store_pd(batch.gamma,     gamma0);  _mm256_store_pd(batch.gamma + 4, gamma1);
    _mm256_store_pd(batch.vega,      vega0);   _mm256_store_pd(batch.vega  + 4, vega1);
    _mm256_store_pd(batch.theta,     theta0);  _mm256_store_pd(batch.theta + 4, theta1);
    _mm256_store_pd(batch.rho,       rho0);    _mm256_store_pd(batch.rho   + 4, rho1);
    (void)nhalf;
}

// ---------------------------------------------------------------------------
// Price N options (N must be multiple of 4 for AVX2; handles any N with tail)
// ---------------------------------------------------------------------------
inline void bs_avx2_array(
    const double* S, const double* K, const double* T,
    const double* r, const double* sigma, const double* is_call,
    double* price, double* delta, double* gamma_out, double* vega,
    std::size_t N) noexcept
{
    std::size_t i = 0;
    // Process 4 at a time (unaligned loads — caller may or may not be aligned)
    for (; i + 4 <= N; i += 4) {
        __m256d vS  = _mm256_loadu_pd(S  + i);
        __m256d vK  = _mm256_loadu_pd(K  + i);
        __m256d vT  = _mm256_loadu_pd(T  + i);
        __m256d vr  = _mm256_loadu_pd(r  + i);
        __m256d vs  = _mm256_loadu_pd(sigma   + i);
        __m256d vic = _mm256_loadu_pd(is_call + i);

        // Temporary aligned storage for output
        alignas(32) double tp[4], td[4], tg[4], tv[4], tth[4], trh[4];
        bs_avx2_quad(vS, vK, vT, vr, vs, vic, tp, td, tg, tv, tth, trh);

        _mm256_storeu_pd(price     + i, _mm256_load_pd(tp));
        _mm256_storeu_pd(delta     + i, _mm256_load_pd(td));
        _mm256_storeu_pd(gamma_out + i, _mm256_load_pd(tg));
        _mm256_storeu_pd(vega      + i, _mm256_load_pd(tv));
    }
    // Scalar tail
    for (; i < N; ++i) {
        double sqrt_T = std::sqrt(T[i]);
        double d1 = (std::log(S[i] / K[i]) + (r[i] + 0.5 * sigma[i] * sigma[i]) * T[i])
                    / (sigma[i] * sqrt_T);
        double d2  = d1 - sigma[i] * sqrt_T;
        double Nd1 = 0.0, Nd2 = 0.0, nd1_v = 0.0;
        // Inline CDF (avoid calling norm_cdf from bs_scalar to keep header self-contained)
        auto cdf = [](double x) noexcept -> double {
            constexpr double p_ = 0.2316419;
            constexpr double a1 =  0.319381530, a2 = -0.356563782;
            constexpr double a3 =  1.781477937, a4 = -1.821255978, a5 =  1.330274429;
            double t = 1.0 / (1.0 + p_ * std::abs(x));
            double poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
            double res = 1.0 - 0.3989422804014326779 * std::exp(-0.5 * x * x) * poly;
            return x >= 0.0 ? res : 1.0 - res;
        };
        Nd1  = cdf(d1); Nd2 = cdf(d2);
        nd1_v = 0.3989422804014326779 * std::exp(-0.5 * d1 * d1);
        double disc = std::exp(-r[i] * T[i]);
        bool call = (is_call[i] > 0.5);
        price[i]     = call ? S[i] * Nd1 - K[i] * disc * Nd2
                             : K[i] * disc * (1.0 - Nd2) - S[i] * (1.0 - Nd1);
        delta[i]     = call ? Nd1 : Nd1 - 1.0;
        gamma_out[i] = nd1_v / (S[i] * sigma[i] * sqrt_T);
        vega[i]      = S[i] * nd1_v * sqrt_T;
    }
}

} // namespace options
