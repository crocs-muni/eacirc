#pragma once

#include <type_traits>

namespace core {

/**
 * @brief implements compile-time sequence of std::size_t
 */

template <std::size_t... xs> struct sequence {
    static constexpr std::size_t size = sizeof...(xs);
};

namespace _impl {

template <std::size_t n, std::size_t... xs>
struct make_sequence : make_sequence<n - 1, n - 1, xs...> {};

template <std::size_t... xs> struct make_sequence<0, xs...> {
    using type = sequence<xs...>;
};

} // namespace _impl

/**
 * @brief helper for creating a sequence of n numbers
 */

template <std::size_t n> using make_sequence = _impl::make_sequence<n>;

/**
 * @brief helper for creating a sequence from parameter pack
 */

template <class... T>
using make_sequence_for = typename make_sequence<sizeof...(T)>::type;

/**
 * @brief variadic logical AND metafunction
 */

template <class...> struct conjunction : std::true_type {};

template <class B, class... Bs>
struct conjunction<B, Bs...>
        : std::conditional_t<B::value, conjunction<Bs...>, B> {};

/**
 * @brief variadic logical OR metafunction
 */

template <class...> struct disjunction : std::false_type {};

template <class B, class... Bs>
struct disjunction<B, Bs...>
        : std::conditional_t<B::value, B, disjunction<Bs...>> {};

/**
 * @brief logical NOT metafunction
 */
template <class B> struct negation : std::integral_constant<bool, !B::value> {};

/**
 * @brief checks element \a E for belonging to the set \a Set
 */

template <class E, class... Set>
struct contains : disjunction<std::is_same<E, Set>...> {};

/**
 * @brief checks whether every element is unique in the given parameter pack
 */

template <class...> struct all_unique : std::true_type {};

template <class T, class... Ts>
struct all_unique<T, Ts...>
        : conjunction<negation<contains<T, Ts...>>, all_unique<Ts...>> {};

/**
 * @brief choose a proper type from several alternatives (the first feasible
 * alternative is chosen)
 */
template <class... Option> struct choose {};

template <class Opt, class... Opts>
struct choose<Opt, Opts...>
        : std::conditional_t<Opt::value, typename Opt::type, choose<Opts...>> {
};

/**
 * @brief alias for \a choose metaclass
 * Usage:
 * template <int value> using feasible_type = choose_t<opt<value == 0, int>
 *                                                     opt<value == 1, char>,
 *                                                     opt<value == 3, void*>>
 * now a \a feasible_type<0> is an alias for \a int and \a feasible_type<3> is
 * an alias for \a void*
 */
template <class... Option> using choose_t = typename choose<Option...>::type;

/**
 * @brief an option metaclass used in \a choose type trait
 */
template <bool B, class T> struct opt : std::integral_constant<bool, B> {
    using type = T;
};

/**
 * @brief variadic metaclass for computing maximum
 */
template <std::size_t...>
struct max : std::integral_constant<std::size_t, 0> {};

template <std::size_t X, std::size_t... Xs>
struct max<X, Xs...>
        : std::conditional_t<(X > max<Xs...>::value),
                             std::integral_constant<std::size_t, X>,
                             max<Xs...>> {};

template <class, template <class...> class>
struct is_specialization_of : std::false_type {};

template <template <class...> class T, class... Args>
struct is_specialization_of<T<Args...>, T> : std::true_type {};

} // namespace core
