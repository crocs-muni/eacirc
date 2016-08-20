#include "eacirc.h"
#include <fstream>

eacirc::eacirc(std::string config)
    : eacirc(std::ifstream(config)) {
}

eacirc::eacirc(std::istream& config)
    : _config(core::json::parse(config))
    , _seed(seed<std::uint64_t>::create(_config["seed"]))
    , _seed_source(_seed.value())
    , _num_of_epochs(_config["num-of-epochs"])
    , _significance_level(_config["significance-level"]) {
}

void eacirc::run() {
    std::vector<double> pvalues;
    pvalues.reserve(_num_of_epochs);

    for (std::size_t i = 0; i != _num_of_epochs; ++i) {
        auto pvalue = _backend->train();

        pvalues.emplace_back(pvalue);
    }

    // TODO: ks_test(pvalues, _significance_level);
}
