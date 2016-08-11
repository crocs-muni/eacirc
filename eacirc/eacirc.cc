#include "eacirc.h"
#include <fstream>

eacirc::eacirc(std::string settings)
    : eacirc(std::ifstream(settings)) {}

eacirc::eacirc(std::istream &settings)
    : _settings(core::json::parse(settings))
    , _seed(seed<std::uint64_t>::create(_settings["main"]["seed"])) {}

void eacirc::run() {}
