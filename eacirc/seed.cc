#include "seed.h"
#include <random>
#include <sstream>

template <typename Type, typename Generator>
static auto generate_one(Generator&& gen) -> std::enable_if_t<std::is_unsigned<Type>::value, Type> {
    return std::uniform_int_distribution<Type>()(gen);
}

template <typename Type>
static auto from_hex(std::string str) -> std::enable_if_t<std::is_unsigned<Type>::value, Type> {
    std::istringstream in(str);
    in.exceptions(std::ios::eofbit | std::ios::failbit | std::ios::badbit);
    in.flags(std::ios_base::hex);

    Type value;
    in >> value;

    return value;
}

template <typename Type>
static auto to_hex(Type value) -> std::enable_if_t<std::is_unsigned<Type>::value, std::string> {
    std::ostringstream out;
    out.exceptions(std::ios::eofbit | std::ios::failbit | std::ios::badbit);
    out.flags(std::ios_base::hex);

    out << value;

    return out.str();
}

seed::seed(std::nullptr_t)
    : _value(generate_one<value_type>(std::random_device())) {}

seed::seed(std::string str)
    : _value(from_hex<value_type>(str)) {}

seed::operator std::string() const {
    return to_hex(_value);
}
