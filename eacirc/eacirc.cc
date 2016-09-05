#include "eacirc.h"
#include "statistics.h"
#include <core/logger.h>
#include <core/random.h>
#include <fstream>
#include <pcg/pcg_random.hpp>

#include "circuit/backend.h"
#include "streams/filestream.h"

static std::ifstream open_config_file(std::string path) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("can't open config file " + path);
    return file;
}

eacirc::eacirc(std::string config)
    : eacirc(open_config_file(config)) {}

eacirc::eacirc(json const& config)
    : _config(config)
    , _seed(seed::create(config["seed"]))
    , _num_of_epochs(config["num-of-epochs"])
    , _significance_level(config["significance-level"])
    , _tv_size(config["tv-size"])
    , _tv_count(config["tv-count"]) {
    seed_seq_from<pcg32> main_seeder(_seed);

    // TODO: create streams & backend
    _stream_a = std::make_unique<streams::filestream>(config["stream-a"]);
    _stream_b = std::make_unique<streams::filestream>(config["stream-b"]);

    _backend = circuit::create_backend(_tv_size, config["backend"], main_seeder);
}

void eacirc::run() {
    std::vector<double> pvalues;
    pvalues.reserve(_num_of_epochs);

    dataset a{_tv_size, _tv_count};
    dataset b{_tv_size, _tv_count};

    _stream_a->read(a);
    _stream_b->read(b);

    for (std::size_t i = 0; i != _num_of_epochs; ++i) {
        _backend->train(a, b);

        _stream_a->read(a);
        _stream_b->read(b);

        pvalues.emplace_back(_backend->test(a, b));

        std::cout << i << std::endl;
    }

    ks_uniformity_test test{pvalues, _significance_level};

    logger::info() << "KS test on p-values of size " << pvalues.size() << std::endl;
    logger::info() << "KS statistics: " << test.test_statistic << std::endl;
    logger::info() << "KS critical value: " << _significance_level << "%: " << test.critical_value
                   << std::endl;

    if (test.test_statistic > test.critical_value) {
        logger::info() << "KS is in " << _significance_level
                       << "% interval -> uniformity hypothesis rejected" << std::endl;
    } else {
        logger::info() << "KS is not in " << _significance_level
                       << "% interval -> uniformity hypothesis not rejected" << std::endl;
    }
}
