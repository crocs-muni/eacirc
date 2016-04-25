#pragma once

#include "genotype.h"
#include <algorithm>
#include <cassert>
#include <core/base.h>
#include <core/compiletime.h>

namespace circuit {
template <class Def, template <class, size_t> class Storage> class Interpreter {
    Storage<u8, Max<Def::in, Def::out>::value> _in;
    Storage<u8, Max<Def::in, Def::out>::value> _out;
    Genotype<Def> const& _circuit;

public:
    Interpreter(Genotype<Def> const& circ) : _circuit(circ) {}

    DataVec<Def::out> operator()(DataVec<Def::in> const& input) {
        std::copy(input.begin(), input.end(), _in.begin());

        for (auto& layer : _circuit) {
            auto out = _out.begin();

            for (auto& node : layer)
                *out++ = execute_node(node);
            std::swap(_in, _out);
        }

        DataVec<Def::out> output;
        std::copy_n(_in.begin(), output.size(), output.begin());
        return output;
    }

protected:
    u8 execute_node(Node<Def> const& node) noexcept {
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
        }
    }
};

} // namespace circuit
