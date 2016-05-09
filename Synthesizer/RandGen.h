#ifndef RANDGEN_H
#define RANDGEN_H

#include <stdexcept>
#include "../core/base.h"
#include "../core/project.h"
#include <random>

template<typename IntType = u64>
class RandGen{
    typedef IntType result_type;
public:
    explicit RandGen(IntType x0 = 1)
    {
        seed(x0);
    }

    template<class It>
    RandGen(It& first, It last)
    {
        seed(first, last);
    }

    result_type static min(){
        return  0;
    }
    result_type max(){
        return  -1;
    }
    // compiler-generated copy constructor and assignment operator are fine
    void seed(IntType x = 1)
    {
        //assert(x >= (min)());
       // assert(x <= (max)());
    }

    template<class It>
    void seed(It& first, It last)
    {
        if(first == last)
            throw std::invalid_argument("Rand::seed");
        seed(*first++);
    }

	virtual IntType operator()()=0;

    // Use a member function; Streamable concept not supported.
    bool operator==(const RandGen& g) const;
    bool operator!=(const RandGen& g) const;

    template<class CharT, class Traits>
    std::basic_ostream<CharT,Traits>&
    operator<<(std::basic_ostream<CharT,Traits>& os);

    template<class CharT, class Traits>
    std::basic_istream<CharT,Traits>&
    operator>>(std::basic_istream<CharT,Traits>& is);
};

#endif //RANDGEN_H
