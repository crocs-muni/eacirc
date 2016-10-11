#include "seed.h"
#include <ios>
#include <random>
#include <sstream>

template <typename Type, typename Generator>
static auto generate_one(Generator&& gen) ->
        typename std::enable_if<std::is_unsigned<Type>::value, Type>::type {
    return std::uniform_int_distribution<Type>()(gen);
}

template <typename Type>
static auto from_hex(std::string str) ->
        typename std::enable_if<std::is_unsigned<Type>::value, Type>::type {
    std::istringstream in(str);

    Type value;
    in >> std::hex >> value;

    if (in.fail())
        throw std::runtime_error("can not convert string to hexadecimal number");

    return value;
}

template <typename Type>
static auto to_hex(Type value) ->
        typename std::enable_if<std::is_unsigned<Type>::value, std::string>::type {
    std::ostringstream out;

    out << std::hex << value;

    if (out.fail())
        throw std::runtime_error("can not convert hexadecimal number to string");

    return out.str();
}

seed::seed(std::nullptr_t)
    : _value(generate_one<value_type>(std::random_device())) {}

seed::seed(std::string str)
    : _value(from_hex<value_type>(str)) {}

seed::operator std::string() const {
    return to_hex(_value);
}
