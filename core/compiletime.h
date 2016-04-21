#pragma once

#include <utility>

/**
 * Compile-time recursive List
 */
template <class...> struct List;
template <> struct List<> {};
template <class T, class... Ts> struct List<T, Ts...> {
    using Head = T;
    using Tail = List<Ts...>;
};

/**
 * Compile-time foldl on List
 */
template <class Fn, class T> T foldl(Fn&&, T&&, List<>) {}
template <class Fn, class T, class... Ts> T foldl(Fn&& fn, T&& value, List<Ts...>) {
    using Head = typename List<Ts...>::Head;
    using Tail = typename List<Ts...>::Tail;

    foldl(std::forward<Fn>(fn), fn(std::forward<T>(value), Head{}), Tail{});
}

/**
  * Compile-time base 2 logarithm, ie Log2<8>::value is 3.
  */
template <unsigned I> struct Log2 { const static unsigned value = 1 + Log2<I / 2>::value; };
template <> struct Log2<1> { const static unsigned value = 1; };

/**
 * Compile-time maximum of the given values, ie. Max<1,9,6>::value is 9.
 */
template <unsigned... I> struct Max;
template <unsigned I> struct Max<I> { const static unsigned value = I; };
template <unsigned I, unsigned... Is> struct Max<I, Is...> {
    const static unsigned value = I > Max<Is...>::value ? I : Max<Is...>::value;
};
