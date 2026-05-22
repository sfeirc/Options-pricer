cmake_minimum_required(VERSION 3.20)
project(options_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -----------------------------------------------------------------------
# Compiler flags
# -----------------------------------------------------------------------
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
    -O3
    -march=native
    -funroll-loops
    -fno-rtti
)

# Optional: AddressSanitizer
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address)
endif()

# Optional: UBSan
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
if(ENABLE_UBSAN)
    add_compile_options(-fsanitize=undefined -g)
    add_link_options(-fsanitize=undefined)
endif()

# -----------------------------------------------------------------------
# Find dependencies
# -----------------------------------------------------------------------
find_package(benchmark QUIET)

# -----------------------------------------------------------------------
# Core library
# -----------------------------------------------------------------------
add_library(options_lib STATIC
    src/options/fd_american.cpp
    src/options/mc_engine.cpp
    src/options/sabr_surface.cpp
    src/options/lm_optimizer.cpp
    src/options/sobol.cpp
)

target_include_directories(options_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# AVX2 + FMA required for SIMD path; AVX-512 is optional
target_compile_options(options_lib PUBLIC
    -mavx2
    -mfma
)

# Suppress specific pedantic warnings for SIMD intrinsic headers
target_compile_options(options_lib PRIVATE
    -Wno-ignored-attributes
)

# -----------------------------------------------------------------------
# Benchmarks (only if Google Benchmark is available)
# -----------------------------------------------------------------------
if(benchmark_FOUND)
    add_executable(bm_options bench/bm_options.cpp)
    target_link_libraries(bm_options PRIVATE options_lib benchmark::benchmark)
    target_compile_options(bm_options PRIVATE -Wno-ignored-attributes)
else()
    message(STATUS "Google Benchmark not found — skipping bm_options target")
    message(STATUS "Install with: apt install libbenchmark-dev  OR  vcpkg install benchmark")
endif()

# -----------------------------------------------------------------------
# Greeks dashboard (ncurses-free ANSI version)
# -----------------------------------------------------------------------
add_executable(greeks_dashboard tools/greeks_dashboard.cpp)
target_link_libraries(greeks_dashboard PRIVATE options_lib)
target_compile_options(greeks_dashboard PRIVATE -Wno-ignored-attributes -Wno-conversion)

# -----------------------------------------------------------------------
# CTest integration
# -----------------------------------------------------------------------
enable_testing()
if(benchmark_FOUND)
    # Run correctness unit tests (UT_ prefix) embedded in the benchmark binary
    add_test(
        NAME unit_avx2_vs_scalar
        COMMAND bm_options --benchmark_filter=UT_AVX2 --benchmark_out_format=console
    )
    add_test(
        NAME unit_fd_american
        COMMAND bm_options --benchmark_filter=UT_FD --benchmark_out_format=console
    )
    add_test(
        NAME unit_sabr_roundtrip
        COMMAND bm_options --benchmark_filter=UT_SABR --benchmark_out_format=console
    )
endif()
