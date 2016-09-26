#pragma once

#include <core/json.h>
#include <core/stream.h>
#include <fstream>

namespace streams {

    struct file_stream : stream {
        file_stream(json const& config)
            : _file(config.at("file").get<std::string>())
            , _in(_file, std::ios::binary) {
            if (!_in.is_open())
                throw std::runtime_error("Cannot open file " + _file);
        }

        void read(dataset& data) override {
            _in.read(reinterpret_cast<char*>(data.data()), std::streamsize(data.size()));

            if (_in.fail())
                throw stream_error("I/O error while reading a file " + _file);
            if (_in.eof())
                throw stream_error("End of file" + _file + " reached, not enough data!");
        }

    private:
        std::string _file;
        std::ifstream _in;
    };

} // namespace streams
