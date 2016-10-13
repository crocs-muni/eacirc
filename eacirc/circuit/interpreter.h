#pragma once

#include "circuit.h"
#include <core/debug.h>
#include <core/view.h>
#include <limits>

namespace circuit {

    template <typename Circuit> struct interpreter {
        using output = typename Circuit::output;

        interpreter(Circuit const& circuit)
            : _circuit(circuit) {}

        template <typename Iterator> output operator()(view<Iterator> in) noexcept {
            ASSERT(in.size() == _circuit.input());
            std::copy(in.begin(), in.end(), _in.begin());

            for (auto const& layer : _circuit) {
                auto o = _out.begin();
                for (auto const& node : layer)
                    *o++ = execute(node);
                std::swap(_in, _out);
            }

            output out;
            std::copy_n(_out.begin(), out.size(), out.begin());
            return out;
        }

    protected:
        std::uint8_t execute(typename Circuit::node const& node) noexcept {
            std::uint8_t result = 0u;

            auto i = node.connectors.iterator();
            const std::uint8_t bits = std::numeric_limits<std::uint8_t>::digits;

            switch (node.function) {
            case fn::NOP:
                if (i.has_next())
                    result = _in[i];
                return result;
            case fn::CONS:
                return node.argument;
            case fn::AND:
                result = 0xff;
                for (; i.has_next(); i.next())
                    result &= _in[i];
                return result;
            case fn::NAND:
                result = 0xff;
                for (; i.has_next(); i.next())
                    result &= _in[i];
                return ~result;
            case fn::OR:
                for (; i.has_next(); i.next())
                    result |= _in[i];
                return result;
            case fn::XOR:
                for (; i.has_next(); i.next())
                    result ^= _in[i];
                return result;
            case fn::NOR:
                for (; i.has_next(); i.next())
                    result |= _in[i];
                return ~result;
            case fn::NOT:
                if (i.has_next())
                    result = ~_in[i];
                return result;
            case fn::SHIL:
                if (i.has_next())
                    result = _in[i] << (node.argument % bits);
                return result;
            case fn::SHIR:
                if (i.has_next())
                    result = _in[i] >> (node.argument % bits);
                return result;
            case fn::ROTL:
                if (i.has_next()) {
                    const std::uint8_t shift = node.argument % bits;
                    if (shift == 0)
                        result = _in[i];
                    else
                        result = (_in[i] << shift) | (_in[i] >> (bits - shift));
                }
                return result;
            case fn::ROTR:
                if (i.has_next()) {
                    const std::uint8_t shift = node.argument % bits;
                    if (shift == 0)
                        result = _in[i];
                    else
                        result = (_in[i] >> shift) | (_in[i] << (bits - shift));
                }
                return result;
            case fn::MASK:
                if (i.has_next())
                    result = _in[i] & node.argument;
                return result;
            case fn::_Size:
                ASSERT_UNREACHABLE();
                return result;
            }
        }

    private:
        vec<Circuit::connectors_type::size> _in;
        vec<Circuit::connectors_type::size> _out;
        Circuit const& _circuit;
    };

} // namespace circuit
