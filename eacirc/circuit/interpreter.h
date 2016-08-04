#pragma once

#include "../dataset.h"
#include "circuit.h"
#include <core/debug.h>
#include <core/traits.h>
#include <limits>

namespace circuit {

template <class Def> struct interpreter {
    using input = datavec<Def::in>;
    using output = datavec<Def::out>;

    interpreter(const circuit<Def> &circuit)
        : _circuit(circuit) {}

    output operator()(const input &in) noexcept {
        _in = {in.begin(), Def::in};

        for (auto &layer : _circuit) {
            auto o = _out.begin();
            for (auto &node : layer)
                *o++ = execute(node);
            std::swap(_in, _out);
        }

        return {_out.begin(), Def::out};
    }

protected:
    std::uint8_t execute(const node<Def> &node) noexcept {
        std::uint8_t result = 0u;

        auto i = node.connectors.begin();
        const auto end = node.connectors.end();
        const std::uint8_t bits = std::numeric_limits<std::uint8_t>::digits;

        switch (node.function) {
        case function::NOP:
            if (i != end)
                result = _in[*i];
            return result;
        case function::CONS:
            return node.argument;
        case function::AND:
            result = 0xff;
            for (; i != end; ++i)
                result &= _in[*i];
            return result;
        case function::NAND:
            result = 0xff;
            for (; i != end; ++i)
                result &= _in[*i];
            return ~result;
        case function::OR:
            for (; i != end; ++i)
                result |= _in[*i];
            return result;
        case function::XOR:
            for (; i != end; ++i)
                result ^= _in[*i];
            return result;
        case function::NOR:
            for (; i != end; ++i)
                result |= _in[*i];
            return ~result;
        case function::NOT:
            if (i != end)
                result = ~_in[*i];
        case function::SHIL:
            if (i != end)
                result = _in[*i] << (node.argument % bits);
            return result;
        case function::SHIR:
            if (i != end)
                result = _in[*i] >> (node.argument % bits);
            return result;
        case function::ROTL:
            if (i != end) {
                const std::uint8_t shift = node.argument % bits;
                if (shift == 0)
                    result = _in[*i];
                else
                    result = (_in[*i] << shift) | (_in[*i] >> (bits - shift));
            }
            return result;
        case function::ROTR:
            if (i != end) {
                const std::uint8_t shift = node.argument % bits;
                if (shift == 0)
                    result = _in[*i];
                else
                    result = (_in[*i] >> shift) | (_in[*i] << (bits - shift));
            }
            return result;
        case function::MASK:
            if (i != end)
                result = _in[*i] & node.argument;
            return result;
        case function::_Size:
            ASSERT_UNREACHABLE();
            return result;
        }
    }

private:
    datavec<core::max<Def::in, Def::out>::value> _in;
    datavec<core::max<Def::in, Def::out>::value> _out;
    const circuit<Def> &_circuit;
};

} // namespace circuit
