#include "settings.h"
#include <minijson_reader.hpp>

using namespace minijson;

istream_context& operator>>(istream_context& ctx, circuit::Settings& settings)
{
    using namespace circuit;

    // clang-format off
    parse_object(ctx, [&](const char* k, value v){
        dispatch(k)
        <<"input_size">> [&]{ settings.input_size = static_cast<unsigned>(v.as_long()); }
        <<"input_size">> [&]{ settings.output_size = static_cast<unsigned>(v.as_long()); }
        <<"function_set">> [&]{
            parse_object(ctx, [&](const char* k, value v){
                dispatch(k)
                <<"NOP">> [&]{ settings.function_set[to_underlying(Function::NOP)] = v.as_bool(); }
                <<"CONS">> [&]{ settings.function_set[to_underlying(Function::CONS)] = v.as_bool(); }
                <<"AND">> [&]{ settings.function_set[to_underlying(Function::AND)] = v.as_bool(); }
                <<"NAND">> [&]{ settings.function_set[to_underlying(Function::NAND)] = v.as_bool(); }
                <<"OR">> [&]{ settings.function_set[to_underlying(Function::OR)] = v.as_bool(); }
                <<"XOR">> [&]{ settings.function_set[to_underlying(Function::XOR)] = v.as_bool(); }
                <<"NOR">> [&]{ settings.function_set[to_underlying(Function::NOR)] = v.as_bool(); }
                <<"NOT">> [&]{ settings.function_set[to_underlying(Function::NOT)] = v.as_bool(); }
                <<"SHIL">> [&]{ settings.function_set[to_underlying(Function::SHIL)] = v.as_bool(); }
                <<"SHIR">> [&]{ settings.function_set[to_underlying(Function::SHIR)] = v.as_bool(); }
                <<"ROTL">> [&]{ settings.function_set[to_underlying(Function::ROTL)] = v.as_bool(); }
                <<"ROTR">> [&]{ settings.function_set[to_underlying(Function::ROTR)] = v.as_bool(); }
                <<"MASK">> [&]{ settings.function_set[to_underlying(Function::MASK)] = v.as_bool(); };
            });
        };
    });
    // clang-format on

    return ctx;
}
