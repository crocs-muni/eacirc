#pragma once

#include <type_traits>

namespace core {

/**
  * @brief negation<B>
  */

template <typename B> struct negation : std::integral_constant<bool, !B::value> {};

/**
 * @brief conjunction<...B>
 */

template <typename...> struct conjunction : std::true_type {};
template <typename B, typename... Bs>
struct conjunction<B, Bs...>
    : std::conditional<B::value, conjunction<Bs...>, std::false_type>::type {};

/**
 * @brief disjunction<...B>
 */

template <typename...> struct disjunction : std::false_type {};
template <typename B, typename... Bs>
struct disjunction<B, Bs...>
    : std::conditional<B::value, std::true_type, disjunction<Bs...>>::type {};

/**
 * @brief is_specialization<Concrete, Abstract>
 */

template <typename, template <typename...> class> struct is_specialization : std::false_type {};

template <template <typename...> class T, typename... Args>
struct is_specialization<T<Args...>, T> : std::true_type {};

/**
 * @brief contains<E , ...Set>
 */

template <typename E, typename... Set> struct contains : disjunction<std::is_same<E, Set>...> {};

/**
 * @brief all_same<...T>
 */

template <typename...> struct all_same : std::true_type {};
template <typename T1, typename T2, typename... Ts>
struct all_same<T1, T2, Ts...>
    : std::conditional<std::is_same<T1, T2>::value, all_same<T2, Ts...>, std::false_type>::type {};

/**
  * @brief all_unique<...T>
  */

template <typename...> struct all_unique : std::true_type {};
template <typename T, typename... Ts>
struct all_unique<T, Ts...> : conjunction<negation<contains<T, Ts...>>, all_unique<Ts...>> {};

/**
  * @brief fst<...T>
  */

template <typename T1, typename...> struct fst { using type = T1; };

/**
* @brief snd<...T>
*/

template <typename T1, typename T2, typename...> struct snd { using type = T2; };

/**
 * @brief max
 */
template <unsigned...> struct max { constexpr static unsigned value = 0; };
template <unsigned V, unsigned... Vs> struct max<V, Vs...> {
  constexpr static unsigned value = (V > max<Vs...>::value) ? V : max<Vs...>::value;
};

} // namespace core
