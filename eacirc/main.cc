#include "eacirc.h"
#include "version.h"
#include <core/cmd.h>
#include <core/logger.h>
#include <limits>

void test_environment() {
    if (std::numeric_limits<std::uint8_t>::max() != 255)
        throw std::range_error("Maximum for unsigned char is not 255");
    if (std::numeric_limits<std::uint8_t>::digits != 8)
        throw std::range_error("Unsigned char does not have 8 bits");
}

static cmd<eacirc::cmd_options> options{{"-h", "--help", "display help message", &eacirc::cmd_options::help},
                           {"-v", "--version", "display program version", &eacirc::cmd_options::version},
                           {"-c", "--config", "specify the config file to load", &eacirc::cmd_options::config},
                           {"-npvals", "--no-pvals", "specify whether not to generate pvals.txt file", &eacirc::cmd_options::not_produce_pvals},
                           {"-nscores", "--no-scores", "specify whether not to generate scores.txt file", &eacirc::cmd_options::not_produce_scores}};

int main(const int argc, const char** argv) try {
    auto cfg = options.parse(make_view(argv, argc));

    if (cfg.help) {
        std::cout << "Usage: eacirc [options]" << std::endl;

        options.print(std::cout);
    } else if (cfg.version) {
        std::cout << "eacirc version " VERSION_TAG << std::endl;
    } else {
        test_environment();

        eacirc app(cfg);
        app.run();
    }

    return 0;
} catch (std::exception& e) {
    logger::error(e.what());
    return 1;
}
