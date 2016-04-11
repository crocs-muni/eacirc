#pragma once

#include "circuit.h"

namespace circuit {
    struct Settings {
        unsigned input_size;
        unsigned output_size;
        std::array<bool, to_underlying(Function::_Size)> function_set;
    };
}

namespace minijson {
    class istream_context;
}

minijson::istream_context& operator>>(minijson::istream_context&,
                                      circuit::Settings&);
