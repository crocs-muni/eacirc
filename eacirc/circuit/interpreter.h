#pragma once

#include "genotype.h"
#include <algorithm>
#include <cassert>
#include <core/base.h>
#include <core/range.h>

namespace circuit {
template <class IO, class Shape, template <class, size_t> class Storage> struct Interpreter {
private:
    Storage<u8, Max<IO::in, IO::out>::value> _in;
    Storage<u8, Max<IO::in, IO::out>::value> _out;
    Genotype<IO, Shape> _circuit;

public:
    Interpreter(Genotype<IO, Shape> const& circuit) : _circuit(circuit) {}

    template <class I, class O> Range<O> operator()(Range<I> input, Range<O> output) noexcept {
        std::copy(input.begin(), input.end(), _in.begin());

        for (auto& layer : _circuit) {
            auto o = _out.begin();
            for (auto& node : layer)
                *o++ = execute(node);
            std::swap(_in, _out);
        }

        std::copy_n(_in.begin(), output.size(), output.begin());
        return output;
    }

protected:
    u8 execute(Node<IO, Shape> const& node) noexcept {
        u8 result = 0u;

        auto i = node.connectors.begin();
        const auto end = node.connectors.end();
        const u8 bits = std::numeric_limits<u8>::digits;

        switch (node.function) {
        case Fn::NOP:
            if (i != end)
                result = _in[*i];
            return result;
        case Fn::CONS:
            return node.argument;
        case Fn::AND:
            result = 0xff;
            for (; i != end; ++i)
                result &= _in[*i];
            return result;
        case Fn::NAND:
            result = 0xff;
            for (; i != end; ++i)
                result &= _in[*i];
            return ~result;
        case Fn::OR:
            for (; i != end; ++i)
                result |= _in[*i];
            return result;
        case Fn::XOR:
            for (; i != end; ++i)
                result ^= _in[*i];
            return result;
        case Fn::NOR:
            for (; i != end; ++i)
                result |= _in[*i];
            return ~result;
        case Fn::NOT:
            if (i != end)
                result = ~_in[*i];
        case Fn::SHIL:
            if (i != end)
                result = _in[*i] << (node.argument % bits);
            return result;
        case Fn::SHIR:
            if (i != end)
                result = _in[*i] >> (node.argument % bits);
            return result;
        case Fn::ROTL:
            if (i != end) {
                const u8 shift = node.argument % bits;
                if (shift == 0)
                    result = _in[*i];
                else
                    result = (_in[*i] << shift) | (_in[*i] >> (bits - shift));
            }
            return result;
        case Fn::ROTR:
            if (i != end) {
                const u8 shift = node.argument % bits;
                if (shift == 0)
                    result = _in[*i];
                else
                    result = (_in[*i] >> shift) | (_in[*i] << (bits - shift));
            }
            return result;
        case Fn::MASK:
            if (i != end)
                result = _in[*i] & node.argument;
            return result;
        case Fn::_Size:
            assert(false);
            return result;
        }
    }
};
} // namespace circuit
