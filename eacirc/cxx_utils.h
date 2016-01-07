#pragma once

// make_unique is defined in MSVS
#ifndef _WIN32

#include <memory>
#include <type_traits>

namespace std {
    template <class T, class... Args>
    typename std::enable_if
    <
        !std::is_array<T>::value,
        std::unique_ptr<T>
    >::type
    make_unique( Args&& ...args )
    {
        return std::unique_ptr<T>( new T( std::forward<Args>(args)... ) );
    }

    template <class T>
    typename std::enable_if
    <
        std::is_array<T>::value,
        std::unique_ptr<T>
    >::type
    make_unique( size_t size )
    {
        using E = typename std::remove_extent<T>::type;
        return unique_ptr<T>( new E[size] );
    }
}

#endif // _WIN32
