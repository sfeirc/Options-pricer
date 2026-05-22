#pragma once
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include "bs_avx2.hpp"  // exp_pd_approx

namespace options {

/// AVX2 batch Crank-Nicolson American option pricer.
/// Prices 4 options simultaneously using __m256d (4 × double).
///
/// Key optimisations vs the scalar pricer:
///   1. SIMD across 4 independent options — each __m256d lane is one option.
///   2. Thomas sweep multipliers (c_prime, inv_m) precomputed outside the time
///      loop: the LHS matrix (a,b,c) is time-independent for constant σ,r, so
///      all N_T×N_S divisions collapse to N_S divisions done once per batch.
///      The hot time-stepping loop uses only FMA + MUL.
///   3. Discount-factor tape: only 2 exp_pd_approx calls per batch (one for
///      exp(-r*dt), one for exp(+r*dt)), then iterative multiply fills N_T slots.
template<int N_S = 50, int N_T = 50>
class FDAmericanPricerAVX2 {
public:
    static_assert(N_S >= 6,  "Need ≥6 nodes for 3-point interpolation");
    static_assert(N_T >= 2,  "Need ≥2 time steps");

    FDAmericanPricerAVX2() noexcept {
        constexpr double eta      = 3.0;
        const     double sinh_eta = std::sinh(eta);
        for (int i = 0; i < N_S; ++i)
            factor_[i] = std::sinh(eta * double(i) / (N_S - 1)) / sinh_eta;
    }

    // Price 4 options simultaneously.
    // S0,K,T,r,sigma: pointers to 4 doubles (need not be aligned).
    // is_call: 4 bools.  out: 4 output prices.
    void price_batch(const double* __restrict__ S0,
                     const double* __restrict__ K,
                     const double* __restrict__ T,
                     const double* __restrict__ r,
                     const double* __restrict__ sigma,
                     const bool*   __restrict__ is_call,
                     double*       __restrict__ out) noexcept {

        const __m256d zero_v = _mm256_setzero_pd();
        const __m256d one_v  = _mm256_set1_pd(1.0);
        const __m256d half_v = _mm256_set1_pd(0.5);
        const __m256d two_v  = _mm256_set1_pd(2.0);
        const __m256d nhalf  = _mm256_set1_pd(-0.5);

        __m256d S0_v     = _mm256_loadu_pd(S0);
        __m256d K_v      = _mm256_loadu_pd(K);
        __m256d T_v      = _mm256_loadu_pd(T);
        __m256d r_v      = _mm256_loadu_pd(r);
        __m256d sigma_v  = _mm256_loadu_pd(sigma);
        __m256d sigma2_v = _mm256_mul_pd(sigma_v, sigma_v);
        __m256d S_max_v  = _mm256_mul_pd(_mm256_set1_pd(4.0), S0_v);
        __m256d dt_v     = _mm256_div_pd(T_v, _mm256_set1_pd(double(N_T)));
        __m256d hdt_v    = _mm256_mul_pd(half_v, dt_v);
        __m256d r_dt_v   = _mm256_mul_pd(r_v, dt_v);
        __m256d hr_dt_v  = _mm256_mul_pd(half_v, r_dt_v);

        // Build call mask (lane = 1.0 → call)
        alignas(32) double call_d[4];
        for (int i = 0; i < 4; ++i) call_d[i] = is_call[i] ? 1.0 : 0.0;
        __m256d cmask = _mm256_cmp_pd(_mm256_loadu_pd(call_d), half_v, _CMP_GT_OQ);

        // ── 1. Spatial grid ────────────────────────────────────────────────────
        for (int i = 0; i < N_S; ++i)
            sg_[i] = _mm256_mul_pd(S_max_v, _mm256_set1_pd(factor_[i]));

        // ── 2. Terminal condition ──────────────────────────────────────────────
        for (int i = 0; i < N_S; ++i) {
            __m256d cp = _mm256_max_pd(_mm256_sub_pd(sg_[i], K_v), zero_v);
            __m256d pp = _mm256_max_pd(_mm256_sub_pd(K_v, sg_[i]), zero_v);
            intr_[i] = V_[i] = _mm256_blendv_pd(pp, cp, cmask);
        }

        // ── 3. Precompute LHS + Thomas multipliers (time-independent) ──────────
        // Boundaries: a=c=0, b=1  →  c_prime=0, inv_m=1
        a_[0]  = zero_v;
        cp_[0] = zero_v;
        im_[0] = one_v;
        rm_[0] = zero_v;
        r0_[0] = one_v;
        rp_[0] = zero_v;

        __m256d prev_cp = zero_v;
        for (int i = 1; i < N_S - 1; ++i) {
            __m256d dSm  = _mm256_sub_pd(sg_[i],     sg_[i - 1]);
            __m256d dSp  = _mm256_sub_pd(sg_[i + 1], sg_[i]);
            __m256d dSav = _mm256_mul_pd(half_v, _mm256_add_pd(dSm, dSp));

            __m256d s2S2 = _mm256_mul_pd(sigma2_v, _mm256_mul_pd(sg_[i], sg_[i]));
            __m256d alp  = _mm256_div_pd(_mm256_mul_pd(hdt_v, s2S2),
                           _mm256_mul_pd(dSm, dSav));
            __m256d bet  = _mm256_div_pd(_mm256_mul_pd(hdt_v, _mm256_mul_pd(r_v, sg_[i])),
                           _mm256_mul_pd(two_v, dSav));

            __m256d ai = _mm256_mul_pd(nhalf, _mm256_sub_pd(alp, bet));
            __m256d bi = _mm256_add_pd(one_v, _mm256_add_pd(alp, hr_dt_v));
            __m256d ci = _mm256_mul_pd(nhalf, _mm256_add_pd(alp, bet));

            // m = b - a*c_prime[i-1]; inv_m = 1/m
            __m256d m_i  = _mm256_fnmadd_pd(ai, prev_cp, bi);
            __m256d im_i = _mm256_div_pd(one_v, m_i);
            __m256d cpi  = _mm256_mul_pd(ci, im_i);

            a_[i]  = ai;
            cp_[i] = cpi;
            im_[i] = im_i;
            prev_cp = cpi;

            // Explicit (RHS) CN coefficients
            rm_[i] = _mm256_mul_pd(half_v, _mm256_sub_pd(alp, bet));
            r0_[i] = _mm256_sub_pd(one_v, _mm256_add_pd(alp, hr_dt_v));
            rp_[i] = _mm256_mul_pd(half_v, _mm256_add_pd(alp, bet));
        }
        const int L = N_S - 1;
        a_[L]  = zero_v;
        cp_[L] = zero_v;
        im_[L] = one_v;
        rm_[L] = zero_v;
        r0_[L] = one_v;
        rp_[L] = zero_v;

        // ── 4. Discount-factor tape (2 exp calls + N_T multiplies) ─────────────
        disc_[N_T - 1] = exp_pd_approx(
            _mm256_mul_pd(_mm256_sub_pd(zero_v, r_v), dt_v));
        __m256d decay  = exp_pd_approx(r_dt_v);
        for (int t = N_T - 2; t >= 0; --t)
            disc_[t] = _mm256_mul_pd(disc_[t + 1], decay);

        // ── 5. Backward time-stepping (hot loop — no divisions) ────────────────
        for (int t = N_T - 1; t >= 0; --t) {
            __m256d KD = _mm256_mul_pd(K_v, disc_[t]);

            // Boundary RHS
            d_[0] = _mm256_blendv_pd(KD,    zero_v,                       cmask);
            d_[L] = _mm256_blendv_pd(zero_v, _mm256_sub_pd(S_max_v, KD),  cmask);

            // Interior RHS (explicit CN)
            for (int i = 1; i < L; ++i)
                d_[i] = _mm256_fmadd_pd(rm_[i], V_[i - 1],
                        _mm256_fmadd_pd(r0_[i], V_[i],
                        _mm256_mul_pd  (rp_[i], V_[i + 1])));

            // Thomas forward sweep: multiply by inv_m (no division)
            dp_[0] = d_[0];
            for (int i = 1; i < N_S; ++i)
                dp_[i] = _mm256_mul_pd(
                    _mm256_fnmadd_pd(a_[i], dp_[i - 1], d_[i]),
                    im_[i]);

            // Back-substitution
            for (int i = N_S - 2; i >= 0; --i)
                dp_[i] = _mm256_fnmadd_pd(cp_[i], dp_[i + 1], dp_[i]);

            // Early exercise
            for (int i = 0; i < N_S; ++i)
                V_[i] = _mm256_max_pd(dp_[i], intr_[i]);
        }

        // ── 6. Interpolate per lane ────────────────────────────────────────────
        for (int lane = 0; lane < 4; ++lane)
            out[lane] = interp_lane(S0[lane], lane);
    }

private:
    // ── Constructor-time constant ──────────────────────────────────────────────
    alignas(64) double factor_[N_S];

    // ── Per-batch vectorised workspace ────────────────────────────────────────
    alignas(32) __m256d sg_  [N_S];   // spatial grid
    alignas(32) __m256d V_   [N_S];   // option value (current time step)
    alignas(32) __m256d intr_[N_S];   // intrinsic (early-exercise payoff)
    alignas(32) __m256d d_   [N_S];   // RHS
    alignas(32) __m256d dp_  [N_S];   // Thomas solution

    // ── Precomputed Thomas multipliers (set each price_batch call) ─────────────
    alignas(32) __m256d a_ [N_S];    // lower diagonal
    alignas(32) __m256d cp_[N_S];    // modified upper diagonal
    alignas(32) __m256d im_[N_S];    // 1 / (b - a*c_prime[i-1])
    alignas(32) __m256d rm_[N_S];    // RHS coeff for V[i-1]
    alignas(32) __m256d r0_[N_S];    // RHS coeff for V[i]
    alignas(32) __m256d rp_[N_S];    // RHS coeff for V[i+1]

    // ── Discount-factor tape ───────────────────────────────────────────────────
    alignas(32) __m256d disc_[N_T];

    double interp_lane(double S, int lane) const noexcept {
        // Binary search for bracket
        int lo = 0, hi = N_S - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) >> 1;
            // sg_[mid] is __m256d; lane-th double = *(double*)&sg_[mid] + lane
            if (reinterpret_cast<const double*>(&sg_[mid])[lane] <= S) lo = mid;
            else                                                         hi = mid;
        }
        if (lo >= N_S - 2) lo = N_S - 3;
        if (lo <  0)       lo = 0;

        // Each __m256d stores 4 lanes interleaved: flat index = node*4 + lane
        const auto* sg = reinterpret_cast<const double*>(sg_);
        const auto* vc = reinterpret_cast<const double*>(V_);
        int i0 = lo, i1 = lo + 1, i2 = lo + 2;
        double x0 = sg[i0*4+lane], x1 = sg[i1*4+lane], x2 = sg[i2*4+lane];
        double v0 = vc[i0*4+lane], v1 = vc[i1*4+lane], v2 = vc[i2*4+lane];
        double L0 = (S-x1)*(S-x2) / ((x0-x1)*(x0-x2));
        double L1 = (S-x0)*(S-x2) / ((x1-x0)*(x1-x2));
        double L2 = (S-x0)*(S-x1) / ((x2-x0)*(x2-x1));
        return std::max(v0*L0 + v1*L1 + v2*L2, 0.0);
    }
};

// ---------------------------------------------------------------------------
// FDAmericanPricerAVX2x8: prices 8 options simultaneously using two __m256d
// per node (lo = options 0-3, hi = options 4-7).
// Same Crank-Nicolson / Thomas algorithm as FDAmericanPricerAVX2.
// Expected throughput: ~2x FDAmericanPricerAVX2 (two independent SIMD streams
// fill both FMA units on modern x86).
// ---------------------------------------------------------------------------
template<int N_S = 50, int N_T = 40>
class FDAmericanPricerAVX2x8 {
public:
    static_assert(N_S >= 6, "Need >=6 nodes for 3-point interpolation");
    static_assert(N_T >= 2, "Need >=2 time steps");

    FDAmericanPricerAVX2x8() noexcept {
        constexpr double eta      = 3.0;
        const     double sinh_eta = std::sinh(eta);
        for (int i = 0; i < N_S; ++i)
            factor_[i] = std::sinh(eta * double(i) / (N_S - 1)) / sinh_eta;
    }

    // Price 8 options simultaneously.
    // S0, K, T, r, sigma: pointers to 8 doubles (need not be aligned).
    // is_call: 8 bools.  out: 8 output prices.
    void price_batch(const double* __restrict__ S0,
                     const double* __restrict__ K,
                     const double* __restrict__ T,
                     const double* __restrict__ r,
                     const double* __restrict__ sigma,
                     const bool*   __restrict__ is_call,
                     double*       __restrict__ out) noexcept {

        const __m256d zero_v = _mm256_setzero_pd();
        const __m256d one_v  = _mm256_set1_pd(1.0);
        const __m256d half_v = _mm256_set1_pd(0.5);
        const __m256d two_v  = _mm256_set1_pd(2.0);
        const __m256d nhalf  = _mm256_set1_pd(-0.5);

        // Load lo (options 0-3) and hi (options 4-7)
        __m256d S0_lo = _mm256_loadu_pd(S0),       S0_hi = _mm256_loadu_pd(S0    + 4);
        __m256d K_lo  = _mm256_loadu_pd(K),         K_hi  = _mm256_loadu_pd(K     + 4);
        __m256d T_lo  = _mm256_loadu_pd(T),         T_hi  = _mm256_loadu_pd(T     + 4);
        __m256d r_lo  = _mm256_loadu_pd(r),         r_hi  = _mm256_loadu_pd(r     + 4);
        __m256d sig_lo = _mm256_loadu_pd(sigma),   sig_hi = _mm256_loadu_pd(sigma + 4);

        __m256d sig2_lo = _mm256_mul_pd(sig_lo, sig_lo);
        __m256d sig2_hi = _mm256_mul_pd(sig_hi, sig_hi);

        __m256d Smax_lo = _mm256_mul_pd(_mm256_set1_pd(4.0), S0_lo);
        __m256d Smax_hi = _mm256_mul_pd(_mm256_set1_pd(4.0), S0_hi);

        __m256d dt_lo  = _mm256_div_pd(T_lo, _mm256_set1_pd(double(N_T)));
        __m256d dt_hi  = _mm256_div_pd(T_hi, _mm256_set1_pd(double(N_T)));
        __m256d hdt_lo = _mm256_mul_pd(half_v, dt_lo);
        __m256d hdt_hi = _mm256_mul_pd(half_v, dt_hi);
        __m256d rdt_lo = _mm256_mul_pd(r_lo, dt_lo);
        __m256d rdt_hi = _mm256_mul_pd(r_hi, dt_hi);
        __m256d hrdt_lo = _mm256_mul_pd(half_v, rdt_lo);
        __m256d hrdt_hi = _mm256_mul_pd(half_v, rdt_hi);

        // Build call masks
        alignas(32) double cd_lo[4], cd_hi[4];
        for (int i = 0; i < 4; ++i) { cd_lo[i] = is_call[i]     ? 1.0 : 0.0; }
        for (int i = 0; i < 4; ++i) { cd_hi[i] = is_call[i + 4] ? 1.0 : 0.0; }
        __m256d cmask_lo = _mm256_cmp_pd(_mm256_loadu_pd(cd_lo), half_v, _CMP_GT_OQ);
        __m256d cmask_hi = _mm256_cmp_pd(_mm256_loadu_pd(cd_hi), half_v, _CMP_GT_OQ);

        // ── 1. Spatial grid ────────────────────────────────────────────────
        for (int i = 0; i < N_S; ++i) {
            __m256d fac = _mm256_set1_pd(factor_[i]);
            sg_lo_[i] = _mm256_mul_pd(Smax_lo, fac);
            sg_hi_[i] = _mm256_mul_pd(Smax_hi, fac);
        }

        // ── 2. Terminal condition ──────────────────────────────────────────
        for (int i = 0; i < N_S; ++i) {
            __m256d cp_lo = _mm256_max_pd(_mm256_sub_pd(sg_lo_[i], K_lo), zero_v);
            __m256d pp_lo = _mm256_max_pd(_mm256_sub_pd(K_lo, sg_lo_[i]), zero_v);
            in_lo_[i] = V_lo_[i] = _mm256_blendv_pd(pp_lo, cp_lo, cmask_lo);

            __m256d cp_hi = _mm256_max_pd(_mm256_sub_pd(sg_hi_[i], K_hi), zero_v);
            __m256d pp_hi = _mm256_max_pd(_mm256_sub_pd(K_hi, sg_hi_[i]), zero_v);
            in_hi_[i] = V_hi_[i] = _mm256_blendv_pd(pp_hi, cp_hi, cmask_hi);
        }

        // ── 3. Precompute LHS + Thomas multipliers ─────────────────────────
        a_lo_[0] = a_hi_[0] = zero_v;
        cp_lo_[0] = cp_hi_[0] = zero_v;
        im_lo_[0] = im_hi_[0] = one_v;
        rm_lo_[0] = rm_hi_[0] = zero_v;
        r0_lo_[0] = r0_hi_[0] = one_v;
        rp_lo_[0] = rp_hi_[0] = zero_v;

        __m256d prev_cp_lo = zero_v, prev_cp_hi = zero_v;
        for (int i = 1; i < N_S - 1; ++i) {
            __m256d dSm_lo = _mm256_sub_pd(sg_lo_[i],     sg_lo_[i - 1]);
            __m256d dSm_hi = _mm256_sub_pd(sg_hi_[i],     sg_hi_[i - 1]);
            __m256d dSp_lo = _mm256_sub_pd(sg_lo_[i + 1], sg_lo_[i]);
            __m256d dSp_hi = _mm256_sub_pd(sg_hi_[i + 1], sg_hi_[i]);
            __m256d dSav_lo = _mm256_mul_pd(half_v, _mm256_add_pd(dSm_lo, dSp_lo));
            __m256d dSav_hi = _mm256_mul_pd(half_v, _mm256_add_pd(dSm_hi, dSp_hi));

            __m256d s2S2_lo = _mm256_mul_pd(sig2_lo, _mm256_mul_pd(sg_lo_[i], sg_lo_[i]));
            __m256d s2S2_hi = _mm256_mul_pd(sig2_hi, _mm256_mul_pd(sg_hi_[i], sg_hi_[i]));

            __m256d alp_lo = _mm256_div_pd(_mm256_mul_pd(hdt_lo, s2S2_lo),
                             _mm256_mul_pd(dSm_lo, dSav_lo));
            __m256d alp_hi = _mm256_div_pd(_mm256_mul_pd(hdt_hi, s2S2_hi),
                             _mm256_mul_pd(dSm_hi, dSav_hi));

            __m256d bet_lo = _mm256_div_pd(_mm256_mul_pd(hdt_lo, _mm256_mul_pd(r_lo, sg_lo_[i])),
                             _mm256_mul_pd(two_v, dSav_lo));
            __m256d bet_hi = _mm256_div_pd(_mm256_mul_pd(hdt_hi, _mm256_mul_pd(r_hi, sg_hi_[i])),
                             _mm256_mul_pd(two_v, dSav_hi));

            __m256d ai_lo = _mm256_mul_pd(nhalf, _mm256_sub_pd(alp_lo, bet_lo));
            __m256d ai_hi = _mm256_mul_pd(nhalf, _mm256_sub_pd(alp_hi, bet_hi));
            __m256d bi_lo = _mm256_add_pd(one_v, _mm256_add_pd(alp_lo, hrdt_lo));
            __m256d bi_hi = _mm256_add_pd(one_v, _mm256_add_pd(alp_hi, hrdt_hi));
            __m256d ci_lo = _mm256_mul_pd(nhalf, _mm256_add_pd(alp_lo, bet_lo));
            __m256d ci_hi = _mm256_mul_pd(nhalf, _mm256_add_pd(alp_hi, bet_hi));

            __m256d mi_lo = _mm256_fnmadd_pd(ai_lo, prev_cp_lo, bi_lo);
            __m256d mi_hi = _mm256_fnmadd_pd(ai_hi, prev_cp_hi, bi_hi);
            __m256d imi_lo = _mm256_div_pd(one_v, mi_lo);
            __m256d imi_hi = _mm256_div_pd(one_v, mi_hi);
            __m256d cpi_lo = _mm256_mul_pd(ci_lo, imi_lo);
            __m256d cpi_hi = _mm256_mul_pd(ci_hi, imi_hi);

            a_lo_[i]  = ai_lo;   a_hi_[i]  = ai_hi;
            cp_lo_[i] = cpi_lo;  cp_hi_[i] = cpi_hi;
            im_lo_[i] = imi_lo;  im_hi_[i] = imi_hi;
            prev_cp_lo = cpi_lo; prev_cp_hi = cpi_hi;

            rm_lo_[i] = _mm256_mul_pd(half_v, _mm256_sub_pd(alp_lo, bet_lo));
            rm_hi_[i] = _mm256_mul_pd(half_v, _mm256_sub_pd(alp_hi, bet_hi));
            r0_lo_[i] = _mm256_sub_pd(one_v, _mm256_add_pd(alp_lo, hrdt_lo));
            r0_hi_[i] = _mm256_sub_pd(one_v, _mm256_add_pd(alp_hi, hrdt_hi));
            rp_lo_[i] = _mm256_mul_pd(half_v, _mm256_add_pd(alp_lo, bet_lo));
            rp_hi_[i] = _mm256_mul_pd(half_v, _mm256_add_pd(alp_hi, bet_hi));
        }
        const int L = N_S - 1;
        a_lo_[L] = a_hi_[L] = zero_v;
        cp_lo_[L] = cp_hi_[L] = zero_v;
        im_lo_[L] = im_hi_[L] = one_v;
        rm_lo_[L] = rm_hi_[L] = zero_v;
        r0_lo_[L] = r0_hi_[L] = one_v;
        rp_lo_[L] = rp_hi_[L] = zero_v;

        // ── 4. Discount-factor tape ────────────────────────────────────────
        disc_lo_[N_T - 1] = exp_pd_approx(_mm256_mul_pd(_mm256_sub_pd(zero_v, r_lo), dt_lo));
        disc_hi_[N_T - 1] = exp_pd_approx(_mm256_mul_pd(_mm256_sub_pd(zero_v, r_hi), dt_hi));
        __m256d decay_lo = exp_pd_approx(rdt_lo);
        __m256d decay_hi = exp_pd_approx(rdt_hi);
        for (int t = N_T - 2; t >= 0; --t) {
            disc_lo_[t] = _mm256_mul_pd(disc_lo_[t + 1], decay_lo);
            disc_hi_[t] = _mm256_mul_pd(disc_hi_[t + 1], decay_hi);
        }

        // ── 5. Backward time-stepping ──────────────────────────────────────
        // d_lo_/d_hi_ arrays are not used here; RHS is computed inline and
        // immediately consumed by the Thomas forward sweep (fused pass).
        for (int t = N_T - 1; t >= 0; --t) {
            __m256d KD_lo = _mm256_mul_pd(K_lo, disc_lo_[t]);
            __m256d KD_hi = _mm256_mul_pd(K_hi, disc_hi_[t]);

            // Boundary: dp[0] = d[0] (boundary nodes: a=0, im=1, so dp = d)
            dp_lo_[0] = _mm256_blendv_pd(KD_lo, zero_v, cmask_lo);
            dp_hi_[0] = _mm256_blendv_pd(KD_hi, zero_v, cmask_hi);

            // Fused RHS build + Thomas forward sweep for interior nodes
            for (int i = 1; i < L; ++i) {
                // RHS: d[i] = rm*V[i-1] + r0*V[i] + rp*V[i+1]
                __m256d di_lo = _mm256_fmadd_pd(rm_lo_[i], V_lo_[i - 1],
                                _mm256_fmadd_pd(r0_lo_[i], V_lo_[i],
                                _mm256_mul_pd  (rp_lo_[i], V_lo_[i + 1])));
                __m256d di_hi = _mm256_fmadd_pd(rm_hi_[i], V_hi_[i - 1],
                                _mm256_fmadd_pd(r0_hi_[i], V_hi_[i],
                                _mm256_mul_pd  (rp_hi_[i], V_hi_[i + 1])));
                // Thomas: dp[i] = (d[i] - a[i]*dp[i-1]) * im[i]
                dp_lo_[i] = _mm256_mul_pd(
                    _mm256_fnmadd_pd(a_lo_[i], dp_lo_[i - 1], di_lo), im_lo_[i]);
                dp_hi_[i] = _mm256_mul_pd(
                    _mm256_fnmadd_pd(a_hi_[i], dp_hi_[i - 1], di_hi), im_hi_[i]);
            }

            // Boundary: dp[L] = d[L] (boundary node: a=0, im=1)
            dp_lo_[L] = _mm256_blendv_pd(zero_v, _mm256_sub_pd(Smax_lo, KD_lo), cmask_lo);
            dp_hi_[L] = _mm256_blendv_pd(zero_v, _mm256_sub_pd(Smax_hi, KD_hi), cmask_hi);

            // Back-substitution
            for (int i = N_S - 2; i >= 0; --i) {
                dp_lo_[i] = _mm256_fnmadd_pd(cp_lo_[i], dp_lo_[i + 1], dp_lo_[i]);
                dp_hi_[i] = _mm256_fnmadd_pd(cp_hi_[i], dp_hi_[i + 1], dp_hi_[i]);
            }

            // Early exercise
            for (int i = 0; i < N_S; ++i) {
                V_lo_[i] = _mm256_max_pd(dp_lo_[i], in_lo_[i]);
                V_hi_[i] = _mm256_max_pd(dp_hi_[i], in_hi_[i]);
            }
        }

        // ── 6. Interpolate per lane ────────────────────────────────────────
        for (int lane = 0; lane < 4; ++lane)
            out[lane]     = interp_lane_lo(S0[lane],     lane);
        for (int lane = 0; lane < 4; ++lane)
            out[lane + 4] = interp_lane_hi(S0[lane + 4], lane);
    }

private:
    alignas(64) double factor_[N_S];

    alignas(32) __m256d sg_lo_[N_S], sg_hi_[N_S];
    alignas(32) __m256d V_lo_ [N_S], V_hi_ [N_S];
    alignas(32) __m256d in_lo_[N_S], in_hi_[N_S];
    alignas(32) __m256d d_lo_ [N_S], d_hi_ [N_S];
    alignas(32) __m256d dp_lo_[N_S], dp_hi_[N_S];

    alignas(32) __m256d a_lo_ [N_S], a_hi_ [N_S];
    alignas(32) __m256d cp_lo_[N_S], cp_hi_[N_S];
    alignas(32) __m256d im_lo_[N_S], im_hi_[N_S];
    alignas(32) __m256d rm_lo_[N_S], rm_hi_[N_S];
    alignas(32) __m256d r0_lo_[N_S], r0_hi_[N_S];
    alignas(32) __m256d rp_lo_[N_S], rp_hi_[N_S];
    alignas(32) __m256d disc_lo_[N_T], disc_hi_[N_T];

    double interp_lane_lo(double S, int lane) const noexcept {
        int lo = 0, hi = N_S - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) >> 1;
            if (reinterpret_cast<const double*>(&sg_lo_[mid])[lane] <= S) lo = mid;
            else                                                            hi = mid;
        }
        if (lo >= N_S - 2) lo = N_S - 3;
        if (lo < 0)        lo = 0;
        const auto* sg = reinterpret_cast<const double*>(sg_lo_);
        const auto* vc = reinterpret_cast<const double*>(V_lo_);
        int i0 = lo, i1 = lo + 1, i2 = lo + 2;
        double x0 = sg[i0*4+lane], x1 = sg[i1*4+lane], x2 = sg[i2*4+lane];
        double v0 = vc[i0*4+lane], v1 = vc[i1*4+lane], v2 = vc[i2*4+lane];
        double L0 = (S-x1)*(S-x2) / ((x0-x1)*(x0-x2));
        double L1 = (S-x0)*(S-x2) / ((x1-x0)*(x1-x2));
        double L2 = (S-x0)*(S-x1) / ((x2-x0)*(x2-x1));
        return std::max(v0*L0 + v1*L1 + v2*L2, 0.0);
    }

    double interp_lane_hi(double S, int lane) const noexcept {
        int lo = 0, hi = N_S - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) >> 1;
            if (reinterpret_cast<const double*>(&sg_hi_[mid])[lane] <= S) lo = mid;
            else                                                            hi = mid;
        }
        if (lo >= N_S - 2) lo = N_S - 3;
        if (lo < 0)        lo = 0;
        const auto* sg = reinterpret_cast<const double*>(sg_hi_);
        const auto* vc = reinterpret_cast<const double*>(V_hi_);
        int i0 = lo, i1 = lo + 1, i2 = lo + 2;
        double x0 = sg[i0*4+lane], x1 = sg[i1*4+lane], x2 = sg[i2*4+lane];
        double v0 = vc[i0*4+lane], v1 = vc[i1*4+lane], v2 = vc[i2*4+lane];
        double L0 = (S-x1)*(S-x2) / ((x0-x1)*(x0-x2));
        double L1 = (S-x0)*(S-x2) / ((x1-x0)*(x1-x2));
        double L2 = (S-x0)*(S-x1) / ((x2-x0)*(x2-x1));
        return std::max(v0*L0 + v1*L1 + v2*L2, 0.0);
    }
};

} // namespace options
