#pragma once

#include <core/json.h>
#include <core/stream.h>
#include <memory>

struct block_interface;

struct block_stream : public stream
{
public:
    block_stream(const json& config, std::size_t osize);
    block_stream(block_stream&&);
    ~block_stream();

    vec_view next() override;

private:
    const std::string _algorithm;
    const std::size_t _round;
    const std::size_t _block_size;

    std::unique_ptr<stream> _source;
    std::unique_ptr<stream> _iv;
    std::unique_ptr<stream> _key;

    std::unique_ptr<block_interface> _encryptor;
    std::unique_ptr<block_interface> _decryptor;

    std::vector<value_type> _data;
};
