#pragma once

#include <string>

struct Settings {
    unsigned num_of_epochs;
    unsigned epoch_duration;
    unsigned significance_level;

    unsigned tv_size;
    unsigned num_of_tvs;
};

template <class> struct json;

template <> struct json<Settings> {
    static void load(const std::string, Settings&);
    static void save(const std::string, Settings const&);
};
