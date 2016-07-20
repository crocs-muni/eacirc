#pragma once

#include "circuit.h"
#include <algorithm>
#include <ea-debug.h>
#include <ea-iterators.h>
#include <ea-traits.h>

namespace ea {
namespace circuit {

template <class Def, template <class, size_t> class Storage>
struct interpreter {
    interpreter(const circuit<Def> &circuit)
        : _circuit(circuit) {}

    template <class I, class O> range<O> operator()(range<I> in, range<O> out) {
        return execute(in, out);
    }

protected:
    template <class I, class O> range<O> execute(range<I> in, range<O> out) {
        std::copy(in.begin(), in.end(), _in.begin());

        for (auto &layer : _circuit) {
            auto o = _out.begin();
            for (auto &node : layer)
                *o++ = execute(node);
            std::swap(_in, _out);
        }

        std::copy_n(_in.begin(), out.size(), out.begin());
        return out;
    }

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
    Storage<std::uint8_t, max<Def::in, Def::out>::value> _in;
    Storage<std::uint8_t, max<Def::in, Def::out>::value> _out;
    const circuit<Def> &_circuit;
};

} // namespace circuit
} // namespace ea
