#pragma once

#include <tuple>

namespace core {
namespace _impl {

template <class T, class F, std::size_t... I>
decltype(auto) map(T &&tuple, F &&function, std::index_sequence<I...>) {
    return std::make_tuple(function(std::get<I>(tuple))...);
}

template <class T, class F, std::size_t... I>
decltype(auto) apply(T &&tupple, F &&function, std::index_sequence<I...>) {
    return function(std::get<I>(tupple)...);
}

} // namespace _impl;

template <class T, class F> decltype(auto) map(T &&tuple, F &&function) {
    return _impl::map(
            std::forward<T>(tuple), std::forward<F>(function),
            std::make_index_sequence<std::tuple_size<std::decay_t<T>>::
                                             value>{});
}

template <class F, class T> decltype(auto) apply(F &&function, T &&tupple) {
    return _impl::apply(
            std::forward<T>(tupple), std::forward<F>(function),
            std::make_index_sequence<std::tuple_size<std::decay_t<T>>::
                                             value>{});
}

} // namespace core
