#pragma once

#include <core/debug.h>
#include <core/json.h>
#include <random>
#include <string>

namespace circuit {

    enum class fn : std::uint8_t {
        NOP,
        CONS,
        AND,
        NAND,
        OR,
        XOR,
        NOR,
        NOT,
        SHIL,
        SHIR,
        ROTL,
        ROTR,
        MASK,
        _Size // this must be the last item of this enum
    };

    std::string to_string(fn func) {
        switch (func) {
        case fn::NOP:
            return "NOP";
        case fn::CONS:
            return "CONS";
        case fn::AND:
            return "AND";
        case fn::NAND:
            return "NAND";
        case fn::OR:
            return "OR";
        case fn::XOR:
            return "XOR";
        case fn::NOR:
            return "NOR";
        case fn::NOT:
            return "NOT";
        case fn::SHIL:
            return "SHIL";
        case fn::SHIR:
            return "SHIR";
        case fn::ROTL:
            return "ROTL";
        case fn::ROTR:
            return "ROTR";
        case fn::MASK:
            return "MASK";
        case fn::_Size:
            break;
        }
        throw std::invalid_argument("such function does not exist");
    }

    fn from_string(std::string str) {
        if (str == to_string(fn::NOP))
            return fn::NOP;
        if (str == to_string(fn::CONS))
            return fn::CONS;
        if (str == to_string(fn::AND))
            return fn::AND;
        if (str == to_string(fn::NAND))
            return fn::NAND;
        if (str == to_string(fn::OR))
            return fn::OR;
        if (str == to_string(fn::XOR))
            return fn::XOR;
        if (str == to_string(fn::NOR))
            return fn::NOR;
        if (str == to_string(fn::NOT))
            return fn::NOT;
        if (str == to_string(fn::SHIL))
            return fn::SHIL;
        if (str == to_string(fn::SHIR))
            return fn::SHIR;
        if (str == to_string(fn::ROTL))
            return fn::ROTL;
        if (str == to_string(fn::ROTR))
            return fn::ROTR;
        if (str == to_string(fn::MASK))
            return fn::MASK;
        throw std::invalid_argument("such function does not exist");
    }

    struct fn_set {
        fn_set(std::initializer_list<fn> samples)
            : _size(samples.size()) {
            ASSERT(_size <= _samples.size());
            std::copy(samples.begin(), samples.end(), _samples.begin());
        }

        fn_set(json const& object)
            : _size(0)
            , _samples{} {
            if (_samples.size() < object.size())
                throw std::runtime_error("more function are listed than possible");

            for (auto& item : object) {
                _samples[_size] = from_string(item);
                ++_size;
            }
        }

        template <typename Generator> fn choose(Generator& g) const {
            std::uniform_int_distribution<std::size_t> dist{0, _size - 1};
            return _samples[dist(g)];
        }

    private:
        std::size_t _size;
        std::array<fn, static_cast<std::size_t>(fn::_Size)> _samples;
    };

} // namespace circuit
