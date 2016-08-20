#pragma once

#include "../dataset.h"
#include "circuit.h"
#include <core/debug.h>
#include <core/traits.h>
#include <limits>

namespace circuit {

template <typename Circuit> struct interpreter {
    using input = vec<Circuit::in>;
    using output = vec<Circuit::out>;

    interpreter(Circuit const& circuit)
        : _circuit(circuit) {
    }

    output operator()(input const& in) noexcept {
        _in = {in.begin(), Circuit::in};

        for (auto& layer : _circuit) {
            auto o = _out.begin();
            for (auto& node : layer)
                *o++ = execute(node);
            std::swap(_in, _out);
        }

        return {_out.begin(), Circuit::out};
    }

protected:
    std::uint8_t execute(typename Circuit::node const& node) noexcept {
        std::uint8_t result = 0u;

        auto i = node.connectors.begin();
        const auto end = node.connectors.end();
        const std::uint8_t bits = std::numeric_limits<std::uint8_t>::digits;

        switch (node.function) {
        case fn::NOP:
            if (i != end)
                result = _in[*i];
            return result;
        case fn::CONS:
            return node.argument;
        case fn::AND:
            result = 0xff;
            for (; i != end; ++i)
                result &= _in[*i];
            return result;
        case fn::NAND:
            result = 0xff;
            for (; i != end; ++i)
                result &= _in[*i];
            return ~result;
        case fn::OR:
            for (; i != end; ++i)
                result |= _in[*i];
            return result;
        case fn::XOR:
            for (; i != end; ++i)
                result ^= _in[*i];
            return result;
        case fn::NOR:
            for (; i != end; ++i)
                result |= _in[*i];
            return ~result;
        case fn::NOT:
            if (i != end)
                result = ~_in[*i];
        case fn::SHIL:
            if (i != end)
                result = _in[*i] << (node.argument % bits);
            return result;
        case fn::SHIR:
            if (i != end)
                result = _in[*i] >> (node.argument % bits);
            return result;
        case fn::ROTL:
            if (i != end) {
                const std::uint8_t shift = node.argument % bits;
                if (shift == 0)
                    result = _in[*i];
                else
                    result = (_in[*i] << shift) | (_in[*i] >> (bits - shift));
            }
            return result;
        case fn::ROTR:
            if (i != end) {
                const std::uint8_t shift = node.argument % bits;
                if (shift == 0)
                    result = _in[*i];
                else
                    result = (_in[*i] >> shift) | (_in[*i] << (bits - shift));
            }
            return result;
        case fn::MASK:
            if (i != end)
                result = _in[*i] & node.argument;
            return result;
        case fn::_Size:
            ASSERT_UNREACHABLE();
            return result;
        }
    }

private:
    vec<core::max<Circuit::in, Circuit::out, Circuit::x>::value> _in;
    vec<core::max<Circuit::in, Circuit::out, Circuit::x>::value> _out;
    Circuit const& _circuit;
};

} // namespace circuit
