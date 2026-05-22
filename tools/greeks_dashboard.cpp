// greeks_dashboard.cpp — Real-time ANSI/ncurses-style Greeks dashboard
// Displays a live option chain with all Greeks, updating as params change
// Usage: ./greeks_dashboard [S=100] [r=0.05] [sigma=0.20] [T=1.0]
//
// Controls:
//   q / ESC  — quit
//   +/-      — adjust spot price ±1
//   s/S      — adjust sigma ±0.01
//   r/R      — adjust rate ±0.001
//   t/T      — adjust time ±0.01
//   c/p      — toggle call/put display
//   SPACE    — pause/unpause

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <termios.h>
#include <unistd.h>
#include <sys/select.h>
#include <csignal>
#include <ctime>
#include <array>

#include "options/bs_scalar.hpp"
#include "options/bs_avx2.hpp"

namespace {

// ANSI escape codes
constexpr const char* ESC_CLEAR   = "\033[2J\033[H";
constexpr const char* ESC_BOLD    = "\033[1m";
constexpr const char* ESC_RESET   = "\033[0m";
constexpr const char* ESC_GREEN   = "\033[32m";
constexpr const char* ESC_RED     = "\033[31m";
constexpr const char* ESC_YELLOW  = "\033[33m";
constexpr const char* ESC_CYAN    = "\033[36m";
constexpr const char* ESC_BLUE    = "\033[34m";
constexpr const char* ESC_MAGENTA = "\033[35m";
constexpr const char* ESC_WHITE   = "\033[37m";
constexpr const char* ESC_DIM     = "\033[2m";

// Terminal raw mode helpers
struct RawMode {
    termios old_termios{};
    bool active = false;

    bool enter() {
        if (tcgetattr(STDIN_FILENO, &old_termios) == -1) return false;
        termios raw = old_termios;
        raw.c_lflag &= ~static_cast<unsigned>(ECHO | ICANON);
        raw.c_cc[VMIN]  = 0;
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
        active = true;
        return true;
    }

    ~RawMode() {
        if (active) tcsetattr(STDIN_FILENO, TCSAFLUSH, &old_termios);
    }
};

// Check if there's a keypress waiting (non-blocking)
int read_key() {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    timeval tv{0, 0};
    if (select(1, &fds, nullptr, nullptr, &tv) > 0) {
        char c;
        if (read(STDIN_FILENO, &c, 1) == 1) return static_cast<int>(static_cast<unsigned char>(c));
    }
    return -1;
}

// Format a double with colour based on sign
void print_coloured(double val, int width, int decimals) {
    if (val > 0.0)      std::printf("%s", ESC_GREEN);
    else if (val < 0.0) std::printf("%s", ESC_RED);
    else                std::printf("%s", ESC_WHITE);
    std::printf("%+*.*f%s", width, decimals, val, ESC_RESET);
}

struct DashboardState {
    double S     = 100.0;
    double r     = 0.05;
    double sigma = 0.20;
    double T     = 1.0;
    bool show_calls = true;
    bool paused     = false;

    // Strike range: S ± 30%, 9 strikes
    static constexpr int N_STRIKES = 13;
    static constexpr double STRIKE_RATIOS[N_STRIKES] = {
        0.70, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.40, 1.50
    };
};

void render(const DashboardState& ds) {
    const char* opt_type = ds.show_calls ? "CALL" : "PUT";
    bool is_call = ds.show_calls;

    std::printf("%s", ESC_CLEAR);

    // Header
    std::printf("%s%s Options Greeks Dashboard%s\n", ESC_BOLD, ESC_CYAN, ESC_RESET);
    std::printf("%s%s%s\n",
        ESC_DIM,
        "======================================================"
        "===========================",
        ESC_RESET);

    // Market parameters
    std::printf("  %sSpot%s  S = %s%-8.2f%s  "
                "%sSigma%s s = %s%-6.3f%s  "
                "%sRate%s  r = %s%-6.4f%s  "
                "%sTime%s  T = %s%-5.3f%sy\n",
        ESC_BOLD, ESC_RESET, ESC_YELLOW, ds.S,     ESC_RESET,
        ESC_BOLD, ESC_RESET, ESC_MAGENTA, ds.sigma, ESC_RESET,
        ESC_BOLD, ESC_RESET, ESC_GREEN,   ds.r,     ESC_RESET,
        ESC_BOLD, ESC_RESET, ESC_BLUE,    ds.T,     ESC_RESET);

    std::printf("  %sType:%s %s%-4s%s   "
                "[+/-] spot  [s/S] sigma  [r/R] rate  [t/T] time  "
                "[c/p] type  [SPACE] pause  [q] quit\n\n",
        ESC_BOLD, ESC_RESET,
        is_call ? ESC_GREEN : ESC_RED, opt_type, ESC_RESET);

    // Column headers
    std::printf("%s  %-6s  %-9s  %-9s  %-9s  %-9s  %-10s  %-10s  %-9s  %-9s%s\n",
        ESC_BOLD,
        "Strike", "Price", "Delta", "Gamma", "Vega",
        "Theta/day", "Rho", "Vanna", "Volga",
        ESC_RESET);
    std::printf("%s%s%s\n",
        ESC_DIM,
        "  ------  ---------  ---------  ---------  ---------  "
        "----------  ----------  ---------  ---------",
        ESC_RESET);

    // Compute and display each strike
    for (int i = 0; i < DashboardState::N_STRIKES; ++i) {
        double K = ds.S * DashboardState::STRIKE_RATIOS[i];
        options::BSResult res = options::bs_price(ds.S, K, ds.T, ds.r, ds.sigma, is_call);

        // Moneyness classification
        double moneyness = ds.S / K;
        const char* atm_marker = "";
        const char* row_color  = ESC_RESET;

        if (std::abs(moneyness - 1.0) < 0.02) {
            atm_marker = " <ATM>";
            row_color  = ESC_BOLD;
        } else if ((is_call && moneyness > 1.0) || (!is_call && moneyness < 1.0)) {
            row_color = ESC_GREEN;  // ITM
        } else {
            row_color = ESC_DIM;    // OTM
        }

        std::printf("%s  %-6.1f  ", row_color, K);
        std::printf("%-9.4f  ", res.price);
        print_coloured(res.delta,  8, 4); std::printf("  ");
        print_coloured(res.gamma,  8, 5); std::printf("  ");
        print_coloured(res.vega / 100.0, 8, 5); std::printf("  "); // per 1% move
        print_coloured(res.theta / 365.0, 9, 5); std::printf("  "); // per calendar day
        print_coloured(res.rho / 100.0, 9, 5); std::printf("  "); // per 1bp
        print_coloured(res.vanna, 8, 5); std::printf("  ");
        print_coloured(res.volga / 100.0, 8, 5);
        std::printf("%s%s\n", atm_marker, ESC_RESET);
    }

    // Footer: portfolio summary for unit position
    std::printf("\n%s%s%s\n",
        ESC_DIM,
        "  -----  ------- Greeks normalized: Delta∈[-1,1], "
        "Vega/Rho per 1%, Theta per day  ------",
        ESC_RESET);

    // Implied moments (approximate from smile)
    std::printf("  %sAtm Vol:%s %.2f%%   "
                "%sAtm Delta:%s %.3f   "
                "%sAtm Gamma:%s %.5f   "
                "%sAtm Vega:%s %.4f\n",
        ESC_BOLD, ESC_RESET, ds.sigma * 100.0,
        ESC_BOLD, ESC_RESET, is_call ? options::norm_cdf(
            (std::log(1.0) + (ds.r + 0.5 * ds.sigma * ds.sigma) * ds.T) / (ds.sigma * std::sqrt(ds.T)))
            : options::norm_cdf(
            (std::log(1.0) + (ds.r + 0.5 * ds.sigma * ds.sigma) * ds.T) / (ds.sigma * std::sqrt(ds.T))) - 1.0,
        ESC_BOLD, ESC_RESET, options::norm_pdf(
            (ds.r + 0.5 * ds.sigma * ds.sigma) * ds.T / (ds.sigma * std::sqrt(ds.T))) /
            (ds.S * ds.sigma * std::sqrt(ds.T)),
        ESC_BOLD, ESC_RESET, ds.S *
            options::norm_pdf((ds.r + 0.5 * ds.sigma * ds.sigma) * ds.T /
                              (ds.sigma * std::sqrt(ds.T))) * std::sqrt(ds.T) / 100.0);

    // P&L for 1σ move
    double atm_price  = options::bs_price(ds.S, ds.S, ds.T, ds.r, ds.sigma, is_call).price;
    double dS         = ds.S * ds.sigma * std::sqrt(1.0 / 252.0);  // 1-day σ move
    double up_price   = options::bs_price(ds.S + dS, ds.S, ds.T, ds.r, ds.sigma, is_call).price;
    double down_price = options::bs_price(ds.S - dS, ds.S, ds.T, ds.r, ds.sigma, is_call).price;

    std::printf("  %sATM price:%s %.4f   "
                "%s1σ up P&L:%s",
        ESC_BOLD, ESC_RESET, atm_price,
        ESC_BOLD, ESC_RESET);
    print_coloured(up_price - atm_price, 7, 4);
    std::printf("   %s1σ dn P&L:%s", ESC_BOLD, ESC_RESET);
    print_coloured(down_price - atm_price, 7, 4);

    if (ds.paused) std::printf("   %s  [PAUSED]%s", ESC_YELLOW, ESC_RESET);
    std::printf("\n");

    std::fflush(stdout);
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    DashboardState ds;

    // Parse command-line overrides
    for (int i = 1; i < argc; ++i) {
        double val = 0.0;
        if (std::sscanf(argv[i], "S=%lf", &val) == 1)     ds.S     = val;
        else if (std::sscanf(argv[i], "r=%lf", &val) == 1) ds.r    = val;
        else if (std::sscanf(argv[i], "sigma=%lf", &val) == 1) ds.sigma = val;
        else if (std::sscanf(argv[i], "T=%lf", &val) == 1) ds.T    = val;
        else if (std::strcmp(argv[i], "put") == 0)  ds.show_calls = false;
        else if (std::strcmp(argv[i], "call") == 0) ds.show_calls = true;
    }

    // Hide cursor
    std::printf("\033[?25l");
    std::fflush(stdout);

    RawMode raw;
    raw.enter();

    // Render loop at ~10 Hz
    bool running = true;
    while (running) {
        if (!ds.paused) render(ds);

        // Process input (non-blocking)
        int key = read_key();
        if (key != -1) {
            switch (key) {
            case 'q': case 'Q': case 27 /* ESC */: running = false; break;
            case '+': case '=': ds.S     += 1.0;   break;
            case '-': case '_': ds.S      = std::max(ds.S - 1.0, 1.0); break;
            case 's':           ds.sigma  = std::max(ds.sigma - 0.01, 0.01); break;
            case 'S':           ds.sigma += 0.01;  break;
            case 'r':           ds.r     -= 0.001; break;
            case 'R':           ds.r     += 0.001; break;
            case 't':           ds.T      = std::max(ds.T - 0.01, 0.01); break;
            case 'T':           ds.T     += 0.01;  break;
            case 'c':           ds.show_calls = true;  break;
            case 'p':           ds.show_calls = false; break;
            case ' ':           ds.paused = !ds.paused; break;
            default: break;
            }
            if (!ds.paused) render(ds);
        }

        // Sleep ~100ms between frames
        struct timespec ts{0, 100000000L};
        nanosleep(&ts, nullptr);
    }

    // Restore cursor, clear screen
    std::printf("\033[?25h");
    std::printf("%s", ESC_CLEAR);
    std::printf("Goodbye.\n");
    return 0;
}
