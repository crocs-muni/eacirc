#pragma once

#include <memory>

namespace std {

    template <typename T, typename... Args>
    auto make_unique(Args&&... args) ->
            typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type {
        return std::unique_ptr<T>{new T(std::forward<Args>(args)...)};
    }

    template <typename T>
    auto make_unique(std::size_t size) ->
            typename std::enable_if<std::is_array<T>::value, std::unique_ptr<T>>::type {
        using E = typename std::remove_extent<T>::type;
        return std::unique_ptr<T>{new E[size]()};
    }

} // namespace std
