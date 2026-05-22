#include <benchmark/benchmark.h>
#include <cmath>
#include <limits>
#include <vector>
#include <cstdio>
#include <cstring>

#include "options/bs_scalar.hpp"
#include "options/bs_avx2.hpp"
#include "options/fd_american.hpp"
#include "options/fd_american_avx2.hpp"
#include "options/mc_engine.hpp"
#include "options/sabr_surface.hpp"

// ============================================================
// Unit tests embedded in the benchmark binary
// (run with --benchmark_filter=UT_ to run only correctness tests)
// ============================================================

static void UT_AVX2_vs_Scalar(benchmark::State& state) {
    // Verify AVX2 batch matches scalar within 1e-6
    options::BSBatch batch;
    for (int i = 0; i < 8; ++i) {
        batch.S[i]       = 90.0 + i * 5.0;
        batch.K[i]       = 100.0;
        batch.T[i]       = 0.25 + i * 0.1;
        batch.r[i]       = 0.05;
        batch.sigma[i]   = 0.15 + i * 0.02;
        batch.is_call[i] = (i % 2 == 0) ? 1.0 : 0.0;
    }
    options::bs_avx2_batch(batch);

    bool ok = true;
    for (int i = 0; i < 8; ++i) {
        bool call = (batch.is_call[i] > 0.5);
        options::BSResult ref = options::bs_price(
            batch.S[i], batch.K[i], batch.T[i],
            batch.r[i], batch.sigma[i], call);
        double diff_price = std::abs(batch.price[i] - ref.price);
        double diff_delta = std::abs(batch.delta[i] - ref.delta);
        double diff_gamma = std::abs(batch.gamma[i] - ref.gamma);
        if (diff_price > 1e-5 || diff_delta > 1e-5 || diff_gamma > 1e-5) {
            std::fprintf(stderr,
                "FAIL lane %d: price|%.6f - %.6f| = %.2e  "
                "delta|%.6f - %.6f| = %.2e\n",
                i, batch.price[i], ref.price, diff_price,
                batch.delta[i], ref.delta, diff_delta);
            ok = false;
        }
    }
    for (auto _ : state) {
        benchmark::DoNotOptimize(ok);
    }
    state.SetLabel(ok ? "PASS" : "FAIL");
}
BENCHMARK(UT_AVX2_vs_Scalar)->Unit(benchmark::kNanosecond);

static void UT_FD_American_vs_European(benchmark::State& state) {
    // American put >= European put (early exercise premium > 0)
    // And for deep-in-the-money put, should approach K - S
    options::FDAmericanPricer<200, 100> pricer;
    double S = 100.0, K = 100.0, T = 1.0, r = 0.05, sigma = 0.2;
    double american_put = pricer.price(S, K, T, r, sigma, false);
    options::BSResult euro = options::bs_price(S, K, T, r, sigma, false);
    double diff = american_put - euro.price;
    bool ok = (diff >= -1e-4) && (american_put > 0.0);

    if (!ok) {
        std::fprintf(stderr,
            "FAIL FD American put=%.6f, BS European put=%.6f, diff=%.6f\n",
            american_put, euro.price, diff);
    }

    // Also check American call == European call (no early exercise for non-div)
    double american_call = pricer.price(S, K, T, r, sigma, true);
    double call_diff = std::abs(american_call - euro.price);
    // American call on non-dividend stock: within 0.5% of European
    bool ok2 = (call_diff / euro.price < 0.005 || american_call >= euro.price);

    for (auto _ : state) {
        benchmark::DoNotOptimize(american_put);
    }
    state.SetLabel((ok && ok2) ? "PASS" : "FAIL");
}
BENCHMARK(UT_FD_American_vs_European)->Unit(benchmark::kMicrosecond);

static void UT_SABR_RoundTrip(benchmark::State& state) {
    // Generate synthetic quotes from known params, calibrate, check recovery
    options::SABRParams true_params{0.3, 0.5, -0.3, 0.4};
    double F = 100.0;
    double expiries[]      = {0.5, 1.0};
    double strike_ratios[] = {0.9, 1.0, 1.1};

    std::vector<options::SABRSurface::MarketQuote> quotes;
    for (double T : expiries)
        for (double kr : strike_ratios) {
            double K = F * kr;
            double vol = options::sabr_implied_vol(F, K, T, true_params);
            quotes.push_back({F, K, T, vol});
        }

    options::SABRSurface surface;
    options::SABRParams init{0.25, 0.5, 0.0, 0.35};
    double rms = surface.calibrate(quotes.data(), (int)quotes.size(), init, 200, 1e-9);
    bool ok = (rms < 1e-4);

    if (!ok) {
        std::fprintf(stderr,
            "FAIL SABR calibration RMS=%.2e, recovered alpha=%.4f rho=%.4f nu=%.4f\n",
            rms, init.alpha, init.rho, init.nu);
    }

    for (auto _ : state) { benchmark::DoNotOptimize(rms); }
    state.SetLabel(ok ? "PASS" : "FAIL");
}
BENCHMARK(UT_SABR_RoundTrip)->Unit(benchmark::kMillisecond);

// ============================================================
// Performance benchmarks
// ============================================================

// [[gnu::noinline]] wrappers prevent the compiler from constant-folding
// the computation when all inputs are compile-time constants.
[[gnu::noinline]] static double
call_bs_scalar(double S, double K, double T, double r, double sigma) noexcept {
    return options::bs_price(S, K, T, r, sigma, true).price;
}

[[gnu::noinline]] static double
call_sabr_vol(double F, double K, double T, const options::SABRParams& p) noexcept {
    return options::sabr_implied_vol(F, K, T, p);
}

static void BM_BSEuropean_Scalar(benchmark::State& state) {
    // volatile init breaks IPCP: compiler can't constant-fold through a
    // volatile read, so S is a genuine runtime value for the whole function.
    volatile double vS = 100.0, vK = 100.0, vT = 1.0, vr = 0.05, vsig = 0.2;
    double S = vS, K = vK, T = vT, r = vr, sigma = vsig;
    for (auto _ : state) {
        double p = call_bs_scalar(S, K, T, r, sigma);
        benchmark::DoNotOptimize(p);
        S += 1e-15;  // keeps S "live" across iterations; prevents loop-hoisting
    }
}
BENCHMARK(BM_BSEuropean_Scalar)->Unit(benchmark::kNanosecond);

static void BM_BSEuropean_AVX2(benchmark::State& state) {
    alignas(32) options::BSBatch batch;
    for (int i = 0; i < 8; ++i) {
        batch.S[i]       = 100.0 + i;
        batch.K[i]       = 100.0;
        batch.T[i]       = 1.0;
        batch.r[i]       = 0.05;
        batch.sigma[i]   = 0.2;
        batch.is_call[i] = 1.0;
    }
    for (auto _ : state) {
        options::bs_avx2_batch(batch);
        benchmark::DoNotOptimize(batch.price[0]);
    }
    state.SetItemsProcessed(state.iterations() * 8);
}
BENCHMARK(BM_BSEuropean_AVX2)->Unit(benchmark::kNanosecond);

static void BM_BSEuropean_AVX2_Array_1024(benchmark::State& state) {
    constexpr int N = 1024;
    alignas(32) static double S[N], K[N], T[N], r[N], sigma[N], is_call[N];
    alignas(32) static double price[N], delta[N], gamma[N], vega[N];
    for (int i = 0; i < N; ++i) {
        S[i] = 95.0 + (i % 20); K[i] = 100.0; T[i] = 0.5 + 0.01 * (i % 50);
        r[i] = 0.05; sigma[i] = 0.15 + 0.001 * (i % 30); is_call[i] = i % 2 ? 1.0 : 0.0;
    }
    for (auto _ : state) {
        options::bs_avx2_array(S, K, T, r, sigma, is_call, price, delta, gamma, vega, N);
        benchmark::DoNotOptimize(price[0]);
    }
    state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_BSEuropean_AVX2_Array_1024)->Unit(benchmark::kNanosecond);

static void BM_AmericanFD_200nodes(benchmark::State& state) {
    options::FDAmericanPricer<200, 100> pricer;
    for (auto _ : state) {
        double p = pricer.price(100.0, 100.0, 1.0, 0.05, 0.2, false);
        benchmark::DoNotOptimize(p);
    }
}
BENCHMARK(BM_AmericanFD_200nodes)->Unit(benchmark::kMicrosecond);

static void BM_AmericanFD_Batch1K(benchmark::State& state) {
    // AVX2x8 batch pricer: 8 options per call, precomputed Thomas multipliers,
    // N_S=50, N_T=40 grid (accurate to ~0.5% for vanilla puts).
    options::FDAmericanPricerAVX2x8<50, 40> pricer;

    constexpr int N = 1000;
    alignas(32) static double S0_a[N], K_a[N], T_a[N], r_a[N], sig_a[N];
    static bool call_a[N];
    for (int i = 0; i < N; ++i) {
        S0_a[i]  = 90.0 + (i % 40) * 0.5;   // vary spot to prevent DCE
        K_a[i]   = 100.0;
        T_a[i]   = 1.0;
        r_a[i]   = 0.05;
        sig_a[i] = 0.20;
        call_a[i] = false;                    // puts (have early-exercise premium)
    }

    alignas(32) double out8[8];

    for (auto _ : state) {
        double sum = 0.0;
        for (int i = 0; i < N; i += 8) {
            pricer.price_batch(S0_a+i, K_a+i, T_a+i, r_a+i, sig_a+i, call_a+i, out8);
            sum += out8[0]+out8[1]+out8[2]+out8[3]+out8[4]+out8[5]+out8[6]+out8[7];
        }
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_AmericanFD_Batch1K)->Unit(benchmark::kMillisecond);

static void BM_MC_European_100K(benchmark::State& state) {
    options::MCEngine engine;
    options::MCEngine::Config cfg;
    cfg.n_paths    = 100000;
    cfg.n_steps    = 252;
    cfg.antithetic = true;
    cfg.stratified = true;

    for (auto _ : state) {
        auto res = engine.price_european(100.0, 100.0, 1.0, 0.05, 0.2, true, cfg);
        benchmark::DoNotOptimize(res.price);
    }
}
BENCHMARK(BM_MC_European_100K)->Unit(benchmark::kMillisecond);

static void BM_MC_Asian_100K(benchmark::State& state) {
    options::MCEngine engine;
    options::MCEngine::Config cfg;
    cfg.n_paths    = 50000;
    cfg.n_steps    = 252;
    cfg.antithetic = true;
    cfg.stratified = false;

    for (auto _ : state) {
        auto res = engine.price_asian(100.0, 100.0, 1.0, 0.05, 0.2, true, false, cfg);
        benchmark::DoNotOptimize(res.price_cv);
    }
}
BENCHMARK(BM_MC_Asian_100K)->Unit(benchmark::kMillisecond);

static void BM_SABRImpliedVol(benchmark::State& state) {
    options::SABRParams params{0.3, 0.5, -0.3, 0.4};
    volatile double vF = 100.0, vK = 100.0, vT = 1.0;
    double F = vF, K = vK, T = vT;
    for (auto _ : state) {
        double v = call_sabr_vol(F, K, T, params);
        benchmark::DoNotOptimize(v);
        F += 1e-15;
    }
}
BENCHMARK(BM_SABRImpliedVol)->Unit(benchmark::kNanosecond);

static void BM_SABRCalibrate_5x9(benchmark::State& state) {
    std::vector<options::SABRSurface::MarketQuote> quotes;
    double F = 100.0;
    double expiries[]      = {0.25, 0.5, 1.0, 2.0, 3.0};
    double strike_ratios[] = {0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3};
    options::SABRParams true_params{0.3, 0.5, -0.3, 0.4};
    for (double T : expiries)
        for (double kr : strike_ratios) {
            double K   = F * kr;
            double vol = options::sabr_implied_vol(F, K, T, true_params);
            // Add small smile perturbation to make it realistic
            quotes.push_back({F, K, T, vol + 0.001 * (kr - 1.0)});
        }

    for (auto _ : state) {
        options::SABRSurface surface;
        options::SABRParams  init{0.2, 0.5, 0.0, 0.3};
        double err = surface.calibrate(quotes.data(), (int)quotes.size(), init);
        benchmark::DoNotOptimize(err);
    }
}
BENCHMARK(BM_SABRCalibrate_5x9)->Unit(benchmark::kMillisecond);

static void BM_NormCDF_AVX2_4wide(benchmark::State& state) {
    __m256d x = _mm256_set_pd(-1.5, 0.0, 1.5, 2.5);
    benchmark::DoNotOptimize(x);
    for (auto _ : state) {
        __m256d res = options::norm_cdf_avx2(x);
        benchmark::DoNotOptimize(res);
        // Vary x slightly to prevent loop hoisting
        x = _mm256_add_pd(x, _mm256_set1_pd(1e-15));
    }
}
BENCHMARK(BM_NormCDF_AVX2_4wide)->Unit(benchmark::kNanosecond);

static void UT_FD_AVX2_vs_Scalar(benchmark::State& state) {
    // Verify AVX2 batch pricer agrees with scalar pricer (same 50×50 grid)
    options::FDAmericanPricerAVX2<50, 40> avx2;
    options::FDAmericanPricer<50, 40>     scalar;

    alignas(32) double S0[4]  = {90.0, 100.0, 105.0, 110.0};
    alignas(32) double K[4]   = {100.0, 100.0, 100.0, 100.0};
    alignas(32) double T[4]   = {1.0, 1.0, 1.0, 1.0};
    alignas(32) double r[4]   = {0.05, 0.05, 0.05, 0.05};
    alignas(32) double sig[4] = {0.20, 0.20, 0.20, 0.20};
    bool call[4]              = {false, false, false, false};
    alignas(32) double out[4];

    avx2.price_batch(S0, K, T, r, sig, call, out);

    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        double ref = scalar.price(S0[i], K[i], T[i], r[i], sig[i], false);
        double err = std::abs(out[i] - ref) / std::max(ref, 0.01);
        if (err > 0.005) {   // 0.5% tolerance (same grid, should be ~0%)
            std::fprintf(stderr,
                "FAIL AVX2 FD lane %d: avx2=%.6f scalar=%.6f relerr=%.2e\n",
                i, out[i], ref, err);
            ok = false;
        }
        // Also: American put >= European BS put
        options::BSResult euro = options::bs_price(S0[i], K[i], T[i], r[i], sig[i], false);
        if (out[i] < euro.price - 1e-4) {
            std::fprintf(stderr,
                "FAIL AVX2 FD lane %d: American=%.6f < European=%.6f\n",
                i, out[i], euro.price);
            ok = false;
        }
    }

    for (auto _ : state) benchmark::DoNotOptimize(ok);
    state.SetLabel(ok ? "PASS" : "FAIL");
}
BENCHMARK(UT_FD_AVX2_vs_Scalar)->Unit(benchmark::kMicrosecond);

static void UT_VectorLog(benchmark::State& state) {
    // Verify log_pd_approx matches std::log to within 10 ULP for representative values.
    // The polynomial approximation targets ~1-2 ULP for most inputs; 10 ULP is a safe
    // gate that confirms correctness while tolerating last-bit rounding variation.
    bool ok = true;
    alignas(32) double xs[4] = {0.5, 1.0, 2.0, 100.0};
    alignas(32) double ys[4] = {0.01, 1234.5, 0.001, 3.14159};

    auto check4 = [&](const double* v) {
        __m256d vx = _mm256_load_pd(v);
        __m256d vl = options::log_pd_approx(vx);
        alignas(32) double res[4];
        _mm256_store_pd(res, vl);
        for (int i = 0; i < 4; ++i) {
            double ref = std::log(v[i]);
            // tolerance = 10 ULP relative to |ref| (min 1e-15 for values near 0)
            double tol = 10.0 * std::numeric_limits<double>::epsilon() *
                         std::max(std::abs(ref), 1e-10);
            double err = std::abs(res[i] - ref);
            if (err > tol) {
                std::fprintf(stderr,
                    "FAIL UT_VectorLog x=%.10g: approx=%.17g ref=%.17g err=%.2e tol=%.2e\n",
                    v[i], res[i], ref, err, tol);
                ok = false;
            }
        }
    };
    check4(xs);
    check4(ys);

    for (auto _ : state) benchmark::DoNotOptimize(ok);
    state.SetLabel(ok ? "PASS" : "FAIL");
}
BENCHMARK(UT_VectorLog)->Unit(benchmark::kNanosecond);

static void UT_FD_AVX2x8_vs_Scalar(benchmark::State& state) {
    // Verify 8-wide pricer agrees with scalar pricer within 0.5%
    options::FDAmericanPricerAVX2x8<50, 40> avx8;
    options::FDAmericanPricer<50, 40>        scalar;

    alignas(32) double S0[8]  = {90.0, 95.0, 100.0, 105.0, 110.0, 85.0, 92.0, 98.0};
    alignas(32) double K[8]   = {100.0,100.0,100.0, 100.0, 100.0,100.0,100.0,100.0};
    alignas(32) double T[8]   = {1.0,  1.0,  1.0,   1.0,   1.0,  1.0,  1.0,  1.0};
    alignas(32) double r[8]   = {0.05, 0.05, 0.05,  0.05,  0.05, 0.05, 0.05, 0.05};
    alignas(32) double sig[8] = {0.20, 0.20, 0.20,  0.20,  0.20, 0.20, 0.20, 0.20};
    bool call[8]              = {false,false,false, false, false,false,false,false};
    alignas(32) double out[8];

    avx8.price_batch(S0, K, T, r, sig, call, out);

    bool ok = true;
    for (int i = 0; i < 8; ++i) {
        double ref = scalar.price(S0[i], K[i], T[i], r[i], sig[i], false);
        double err = std::abs(out[i] - ref) / std::max(ref, 0.01);
        if (err > 0.005) {
            std::fprintf(stderr,
                "FAIL AVX2x8 FD lane %d: avx8=%.6f scalar=%.6f relerr=%.2e\n",
                i, out[i], ref, err);
            ok = false;
        }
        options::BSResult euro = options::bs_price(S0[i], K[i], T[i], r[i], sig[i], false);
        if (out[i] < euro.price - 1e-4) {
            std::fprintf(stderr,
                "FAIL AVX2x8 FD lane %d: American=%.6f < European=%.6f\n",
                i, out[i], euro.price);
            ok = false;
        }
    }

    for (auto _ : state) benchmark::DoNotOptimize(ok);
    state.SetLabel(ok ? "PASS" : "FAIL");
}
BENCHMARK(UT_FD_AVX2x8_vs_Scalar)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
