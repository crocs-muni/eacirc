#pragma once

#include "circuit.h"
#include <algorithm>
#include <core/base.h>
#include <memory>

namespace circuit {
    template <class Allocator = std::allocator<ui8>>
    class Interpreter : Allocator {
        unsigned _isize;
        unsigned _osize;
        Owner<ui8*> _in;
        ui8* _out;

    public:
        Interpreter(unsigned in_size, unsigned out_size)
                : _isize(in_size), _osize(out_size),
                  _in(Allocator::allocate(2 * std::max(_isize, _osize))),
                  _out(_in + std::max(_isize, _osize))
        {
        }

        ~Interpreter()
        {
            Allocator::dealocate(_in, 2 * std::max(_isize, _osize));
        }

        template <class Circuit>
        void operator()(Circuit const& circuit, ui8 const* input, ui8* output)
        {
            std::copy(input, input + _isize, _in);

            for (auto layer : circuit.layers()) {
                for (unsigned i = 0; i != Circuit::x; ++i)
                    _out[i] = execute_node(layer[i]);
                std::swap(_in, _out);
            }

            std::copy(_out, _out + _osize, output);
        }

    protected:
        ui8 execute_node(Node const& node) noexcept
        {
            ui8 result = 0u;

            auto i = node.connectors.begin();
            const auto end = node.connectors.end();
            const ui8 bits = std::numeric_limits<ui8>::digits;

            switch (node.function) {
            case Function::NOP:
                if (i != end)
                    result = _in[*i];
                return result;
            case Function::CONS:
                return node.argument;
            case Function::AND:
                result = 0xff;
                for (; i != end; ++i)
                    result &= _in[*i];
                return result;
            case Function::NAND:
                result = 0xff;
                for (; i != end; ++i)
                    result &= _in[*i];
                return ~result;
            case Function::OR:
                for (; i != end; ++i)
                    result |= _in[*i];
                return result;
            case Function::XOR:
                for (; i != end; ++i)
                    result ^= _in[*i];
                return result;
            case Function::NOR:
                for (; i != end; ++i)
                    result |= _in[*i];
                return ~result;
            case Function::NOT:
                if (i != end)
                    result = ~_in[*i];
            case Function::SHIL:
                if (i != end)
                    result = _in[*i] << (node.argument % bits);
                return result;
            case Function::SHIR:
                if (i != end)
                    result = _in[*i] >> (node.argument % bits);
                return result;
            case Function::ROTL:
                if (i != end) {
                    const ui8 shift = node.argument % bits;
                    if (shift == 0)
                        result = _in[*i];
                    else
                        result = (_in[*i] << shift) |
                                 (_in[*i] >> (bits - shift));
                }
                return result;
            case Function::ROTR:
                if (i != end) {
                    const ui8 shift = node.argument % bits;
                    if (shift == 0)
                        result = _in[*i];
                    else
                        result = (_in[*i] >> shift) |
                                 (_in[*i] << (bits - shift));
                }
                return result;
            case Function::MASK:
                if (i != end)
                    result = _in[*i] & node.argument;
                return result;
            }
        }
    };
}
