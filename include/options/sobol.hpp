#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>

namespace options {

// ---------------------------------------------------------------------------
// Sobol quasi-random sequence generator
// Uses Joe-Kuo direction numbers (2010), first 21 dimensions
// Gray-code enumeration for efficiency
// ---------------------------------------------------------------------------
class SobolGenerator {
public:
    static constexpr int MAX_DIM = 21;
    static constexpr int MAX_BITS = 32;

    explicit SobolGenerator(int dimension = 1) : dim_(dimension) {
        reset();
    }

    void reset() noexcept {
        count_ = 0;
        std::memset(state_, 0, sizeof(state_));
        init_direction_numbers();
    }

    // Fill out[0..n_dims-1] with next Sobol point in each dimension
    // Each call advances the sequence by one point
    void next(double* out, int n_dims) noexcept {
        int ndims = n_dims < dim_ ? n_dims : dim_;
        // Determine rightmost zero bit of count
        uint32_t c = count_;
        int bit = 0;
        if (c == 0) {
            bit = 0;
        } else {
            uint32_t tmp = c;
            while (tmp & 1u) { tmp >>= 1; ++bit; }
        }

        constexpr double scale = 1.0 / static_cast<double>(1ULL << MAX_BITS);
        for (int d = 0; d < ndims; ++d) {
            state_[d] ^= v_[d][bit];
            out[d] = static_cast<double>(state_[d]) * scale;
        }
        ++count_;
    }

    // Generate n_points points, each n_dims-dimensional
    // out must have space for n_points * n_dims doubles
    // Layout: out[point * n_dims + dim]
    void generate(double* out, int n_points, int n_dims) noexcept {
        for (int i = 0; i < n_points; ++i) {
            next(out + i * n_dims, n_dims);
        }
    }

private:
    int dim_;
    uint32_t count_;
    uint32_t state_[MAX_DIM];
    uint32_t v_[MAX_DIM][MAX_BITS];  // direction numbers

    void init_direction_numbers() noexcept {
        // Direction numbers from Joe & Kuo (2010), first 21 dimensions
        // Dimension 0 (first dimension) uses standard Sobol
        // v[d][i] = direction number, scaled to 32-bit integer
        // Each v_[d][i] = m_i * 2^(32-i)

        // Primitive polynomials and initial direction numbers
        // Format: {s, a, m[1..s]} where s=degree, a=poly (without leading 1)
        struct DimInit {
            int s;
            uint32_t a;
            uint32_t m[8];
        };

        // Joe-Kuo table for dims 1..20 (dim 0 is identity)
        static const DimInit inits[MAX_DIM - 1] = {
            {1, 0,       {1}},
            {2, 1,       {1, 1}},
            {3, 1,       {1, 1, 1}},
            {3, 2,       {1, 3, 7}},
            {4, 1,       {1, 1, 5, 13}},
            {4, 4,       {1, 3, 1, 1}},
            {5, 2,       {1, 1, 1, 15, 11}},
            {5, 4,       {1, 3, 5, 11, 22}},  // Padded: 22 is a known value
            {5, 7,       {1, 1, 7, 3, 19}},
            {5, 11,      {1, 1, 5, 5, 1}},
            {5, 13,      {1, 1, 1, 15, 17}},
            {5, 14,      {1, 3, 7, 9, 5}},
            {6, 1,       {1, 3, 5, 9, 15, 3}},
            {6, 13,      {1, 1, 7, 13, 27, 51}},  // Padded
            {6, 16,      {1, 1, 3, 7, 27, 21}},   // Padded
            {6, 19,      {1, 3, 3, 15, 29, 15}},  // Padded
            {6, 22,      {1, 1, 5, 11, 23, 9}},   // Padded
            {6, 25,      {1, 3, 5, 11, 13, 3}},   // Padded
            {7, 1,       {1, 1, 1, 3, 17, 27, 45}}, // Padded
            {7, 4,       {1, 1, 1, 5, 29, 35, 7}},  // Padded
        };

        // Dimension 0: v_[0][i] = 2^(31-i)
        for (int i = 0; i < MAX_BITS; ++i) {
            v_[0][i] = 1u << (MAX_BITS - 1 - i);
        }

        // Remaining dimensions
        for (int d = 1; d < MAX_DIM; ++d) {
            const DimInit& init = inits[d - 1];
            int s = init.s;

            // Initial direction numbers
            for (int i = 0; i < s && i < MAX_BITS; ++i) {
                v_[d][i] = init.m[i] << (MAX_BITS - 1 - i);
            }

            // Recurrence: v[i] = v[i-s] XOR (v[i-s] >> s) XOR ...
            for (int i = s; i < MAX_BITS; ++i) {
                v_[d][i] = v_[d][i - s] ^ (v_[d][i - s] >> s);
                for (int k = 1; k < s; ++k) {
                    if ((init.a >> (s - 1 - k)) & 1u) {
                        v_[d][i] ^= v_[d][i - k];
                    }
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Normal quantile transform using Beasley-Springer-Moro algorithm
// Converts uniform [0,1] to standard normal
// ---------------------------------------------------------------------------
inline double normal_quantile(double u) noexcept {
    // Rational approximation for the normal quantile (Beasley-Springer-Moro)
    constexpr double a0 =  2.50662823884;
    constexpr double a1 = -18.61500062529;
    constexpr double a2 =  41.39119773534;
    constexpr double a3 = -25.44106049637;
    constexpr double b0 = -8.47351093090;
    constexpr double b1 =  23.08336743743;
    constexpr double b2 = -21.06224101826;
    constexpr double b3 =  3.13082909833;
    constexpr double c0 =  0.3374754822726147;
    constexpr double c1 =  0.9761690190917186;
    constexpr double c2 =  0.1607979714918209;
    constexpr double c3 =  0.0276438810333863;
    constexpr double c4 =  0.0038405729373609;
    constexpr double c5 =  0.0003951896511349;
    constexpr double c6 =  0.0000321767881768;
    constexpr double c7 =  0.0000002888167364;
    constexpr double c8 =  0.0000003960315187;

    double y = u - 0.5;
    if (std::abs(y) < 0.42) {
        double r = y * y;
        return y * (((a3 * r + a2) * r + a1) * r + a0) /
               ((((b3 * r + b2) * r + b1) * r + b0) * r + 1.0);
    }

    double r = y < 0.0 ? u : 1.0 - u;
    r = std::log(-std::log(r));
    double x = c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 +
               r * (c5 + r * (c6 + r * (c7 + r * c8)))))));
    return y < 0.0 ? -x : x;
}

} // namespace options
