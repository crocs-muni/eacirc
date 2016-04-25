#pragma once

/**
 * Compile-time maximum of the given values, ie. Max<1,9,6>::value is 9.
 */
template <unsigned... I> struct Max;
template <unsigned I> struct Max<I> { const static unsigned value = I; };
template <unsigned I, unsigned... Is> struct Max<I, Is...> {
    const static unsigned value = I > Max<Is...>::value ? I : Max<Is...>::value;
};
