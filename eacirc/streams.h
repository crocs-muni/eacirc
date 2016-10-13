#pragma once

#include <core/json.h>
#include <core/random.h>
#include <core/stream.h>
#include <memory>

std::unique_ptr<stream> make_stream(json const& config, default_seed_source& seeder);
