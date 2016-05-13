#include "settings.h"
#include <core/exceptions.h>
#include <core/logger.h>
#include <fstream>
#include <json.hpp>

void json<Settings>::load(const std::string file, Settings& settings) {
    std::ifstream ifs(file);

    nlohmann::json j;
    ifs >> j;

    try {
        settings.epoch_duration = j["epoch_duration"];
        settings.num_of_epochs = j["num_of_epochs"];
        settings.significance_level = j["significance_level"];
    } catch (std::exception& e) {
        Logger::error() << "Parsing of " << file << " failed with message: " << e.what()
                        << std::endl;
        throw fatal_error();
    }
}

void json<Settings>::save(const std::string file, Settings const& settings) {
    nlohmann::json j;
    j["epoch_duration"] = settings.epoch_duration;
    j["num_of_epochs"] = settings.num_of_epochs;
    j["significance_level"] = settings.significance_level;

    std::ofstream ofs(file);
    ofs << j;
}
