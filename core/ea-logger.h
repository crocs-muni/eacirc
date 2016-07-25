#pragma once

#include "ea-debug.h"
#include <fstream>
#include <iostream>
#include <ostream>
#include <streambuf>

namespace ea {

template <typename char_type, typename traits = std::char_traits<char_type>>
struct basic_teebuf : std::basic_streambuf<char_type, traits> {
public:
    using int_type = typename traits::int_type;

    basic_teebuf(std::basic_streambuf<char_type, traits> *sb1,
                 std::basic_streambuf<char_type, traits> *sb2)
        : sb1(sb1)
        , sb2(sb2) {}

private:
    virtual int sync() {
        int const r1 = sb1->pubsync();
        int const r2 = sb2->pubsync();
        return r1 == 0 && r2 == 0 ? 0 : -1;
    }

    virtual int_type overflow(int_type c) {
        int_type const eof = traits::eof();

        if (traits::eq_int_type(c, eof)) {
            return traits::not_eof(c);
        } else {
            char_type const ch = traits::to_char_type(c);
            int_type const r1 = sb1->sputc(ch);
            int_type const r2 = sb2->sputc(ch);

            return traits::eq_int_type(r1, eof) || traits::eq_int_type(r2, eof)
                           ? eof
                           : c;
        }
    }

private:
    std::basic_streambuf<char_type, traits> *sb1;
    std::basic_streambuf<char_type, traits> *sb2;
};

/**
 * @brief a stream mimicking the functionality of unix utility \a tee
 */
template <typename char_type, typename traits = std::char_traits<char_type>>
struct basic_teestream : std::basic_ostream<char_type, traits> {
public:
    basic_teestream(std::basic_ostream<char_type, traits> &o1,
                    std::basic_ostream<char_type, traits> &o2)
        : std::basic_ostream<char_type, traits>(&tbuf)
        , tbuf(o1.rdbuf(), o2.rdbuf()) {}

private:
    basic_teebuf<char_type, traits> tbuf;
};

using teestream = basic_teestream<char>;
using wteestream = basic_teestream<wchar_t>;

struct logger {
    logger(const char *file)
        : _file(file)
        , _tee(_file, std::cout) {
        ASSERT(instance == nullptr);
        logger::instance = this;
    }

    ~logger() { logger::instance = nullptr; }

    static std::ostream &entry();

protected:
    static logger &get() {
        ASSERT_ALLWAYS(logger::instance != nullptr);
        return *logger::instance;
    }

private:
    std::ofstream _file;
    teestream _tee;

    static logger *instance;
};

#define LOG_ERROR(message_)                                                    \
    logger::entry() << "[error]   " << message_ << std::endl;

#define LOG_WARNING(message_)                                                  \
    logger::entry() << "[warning] " << message_ << std::endl;

} // namespace ea
