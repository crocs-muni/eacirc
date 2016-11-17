#pragma once

#include <core/json.h>
#include <core/random.h>
#include <core/stream.h>
#include <memory>

std::unique_ptr<stream> make_stream(const json& config, default_seed_source& seeder, std::size_t osize);
std::unique_ptr<stream> make_stream(const json& config, std::size_t osize);
void stream_to_dataset(dataset &set, std::unique_ptr<stream>& source);
