#include <ostream>
#include <streambuf>

template <typename char_type, typename traits = std::char_traits<char_type>>
class BasicTeebuf : public std::basic_streambuf<char_type, traits> {
public:
    using int_type = typename traits::int_type;

    BasicTeebuf(std::basic_streambuf<char_type, traits>* sb1,
                std::basic_streambuf<char_type, traits>* sb2)
            : sb1(sb1), sb2(sb2)
    {
    }

private:
    virtual int sync()
    {
        int const r1 = sb1->pubsync();
        int const r2 = sb2->pubsync();
        return r1 == 0 && r2 == 0 ? 0 : -1;
    }

    virtual int_type overflow(int_type c)
    {
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
    std::basic_streambuf<char_type, traits>* sb1;
    std::basic_streambuf<char_type, traits>* sb2;
};

template <typename char_type, typename traits = std::char_traits<char_type>>
class BasicTeestream : public std::basic_ostream<char_type, traits> {
public:
    BasicTeestream(std::basic_ostream<char_type, traits>& o1,
                   std::basic_ostream<char_type, traits>& o2)
            : std::basic_ostream<char_type, traits>(&tbuf),
              tbuf(o1.rdbuf(), o2.rdbuf())
    {
    }

private:
    BasicTeebuf<char_type, traits> tbuf;
};

using Teestream = BasicTeestream<char>;
