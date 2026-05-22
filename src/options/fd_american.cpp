#include "options/fd_american.hpp"
// All implementation is in the header (template class, fully inlined).
// Explicit instantiation for common sizes to avoid link-time issues.
namespace options {
    template class FDAmericanPricer<200, 100>;
    template class FDAmericanPricer<100, 50>;
    template class FDAmericanPricer<400, 200>;
} // namespace options
