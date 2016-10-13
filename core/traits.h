#pragma once

#include <type_traits>

/**
 * @brief variadic logical AND metafunction
 */

template <typename...> struct conjunction : std::true_type {};
template <typename B, typename... Bs>
struct conjunction<B, Bs...> : std::conditional<B::value, conjunction<Bs...>, B>::type {};

/**
 * @brief variadic logical OR metafunction
 */

template <typename...> struct disjunction : std::false_type {};
template <typename B, typename... Bs>
struct disjunction<B, Bs...> : std::conditional<B::value, B, disjunction<Bs...>>::type {};

/**
 * @brief logical NOT metafunction
 */
template <typename B> struct negation : std::integral_constant<bool, !B::value> {};

/**
 * @brief checks element \a E for belonging to the set \a Set
 */

template <typename E, typename... Set> struct contains : disjunction<std::is_same<E, Set>...> {};

/**
 * @brief checks whether every element is unique in the given parameter pack
 */

template <typename...> struct all_unique : std::true_type {};
template <typename T, typename... Ts>
struct all_unique<T, Ts...> : conjunction<negation<contains<T, Ts...>>, all_unique<Ts...>> {};
