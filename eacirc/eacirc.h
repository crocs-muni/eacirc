#pragma once

#include "backend.h"
#include <eacirc-core/seed.h>
#include <eacirc-core/json.h>
#include <eacirc-streams/stream.h>
#include <memory>

struct eacirc {

    struct cmd_options {
        bool help = false;
        bool version = false;
        bool not_produce_pvals= false;
        bool not_produce_scores = false;
        std::string config = "config.json";
    };

    eacirc(cmd_options options);

    eacirc(cmd_options options, std::istream& config)
        : eacirc(options, json::parse(config)) {}

    eacirc(cmd_options options, std::istream&& config)
        : eacirc(options, json::parse(config)) {}

    eacirc(cmd_options options, json&& config)
        : eacirc(options, config) {}

    eacirc(cmd_options options, json const& config);

    void run();

private:
    const cmd_options _cmd_options;
    const json _config_file;
    const seed _seed;

    const std::uint64_t _num_of_epochs;
    const unsigned _significance_level;
    const unsigned _tv_size;
    const std::uint64_t _tv_count;

    std::unique_ptr<backend> _backend;
    std::unique_ptr<stream> _stream_a;
    std::unique_ptr<stream> _stream_b;
};
