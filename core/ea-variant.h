#pragma once

#include "ea-traits.h"
#include <exception>
#include <utility>

namespace ea {
namespace _impl {

template <class...> struct variant;

template <> struct variant<> {
    static constexpr unsigned index = 0;
    template <class Q> static constexpr unsigned index_of() { return 0; }

    static void destruct(const unsigned, void *) {}
    static void move_construct(const unsigned, void *, void *) {}
    static void copy_construct(const unsigned, void *, const void *) {}

    static void move_assign_same(const unsigned, void *, void *) {}
    static void copy_assign_same(const unsigned, void *, const void *) {}

    static void swap_same(const unsigned, void *, void *) {}
};

template <class T, class... Ts> struct variant<T, Ts...> {
    static constexpr unsigned index = 1 + variant<Ts...>::index;

    template <class Q> static constexpr unsigned index_of() {
        return std::is_same<Q, T>::value
                       ? index
                       : variant<Ts...>::template index_of<Q>();
    }

    static void destruct(const unsigned i, void *data) {
        if (i == index)
            ref(data).~T();
        else
            variant<Ts...>::destruct(i, data);
    }

    static void move_construct(unsigned i, void *data, void *other) {
        if (i == index)
            new (data) T(move(other));
        else
            variant<Ts...>::move_construct(i, data, other);
    }

    static void copy_construct(const unsigned i, void *data,
                               const void *other) {
        if (i == index)
            new (data) T(ref(other));
        else
            variant<Ts...>::copy_construct(i, data, other);
    }

    static void move_assign_same(const unsigned i, void *lhs, void *rhs) {
        if (i == index)
            ref(lhs) = move(rhs);
        else
            variant<Ts...>::move_assign_same(i, lhs, rhs);
    }

    static void copy_assign_same(const unsigned i, void *lhs, const void *rhs) {
        if (i == index)
            ref(lhs) = ref(rhs);
        else
            variant<Ts...>::copy_assign_same(i, lhs, rhs);
    }

    static void swap_same(const unsigned i, void *lhs, void *rhs) {
        using std::swap;

        if (i == index)
            swap(ref(lhs), ref(rhs));
        else
            variant<Ts...>::swap_same(i, lhs, rhs);
    }

    static T &&move(void *data) { return std::move(ref(data)); }

    static T &ref(void *data) { return *reinterpret_cast<T *>(data); }

    static const T &ref(const void *data) {
        return *reinterpret_cast<const T *>(data);
    }
};

} // namespace _impl

struct bad_variant_access : std::exception {};

template <class... Types> struct variant {
    static_assert(all_unique<Types...>::value,
                  "Every type in variadic template must be unique.");

    variant()
        : _index(_impl::variant<>::index) {}

    variant(variant &&o)
        : _index(o._index) {
        helper::move_construct(_index, &_data, &o._data);
    }

    variant(const variant &o)
        : _index(o._index) {
        helper::copy_construct(_index, &_data, &o._data);
    }

    template <class T, class U = std::decay_t<T>,
              class = std::enable_if_t<contains<U, Types...>::value>>
    variant(T &&val)
        : _index(helper::template index_of<U>()) {
        new (&_data) U(std::forward<T>(val));
    }

    ~variant() { helper::destruct(_index, &_data); }

    variant &operator=(variant &&o) {
        if (_index == o._index)
            helper::move_assign_same(_index, &_data, &o._data);
        else {
            helper::destruct(_index, &_data);
            helper::move_construct(o._index, &_data, &o._data);
            _index = o._index;
        }
        return *this;
    }

    variant &operator=(const variant &o) {
        if (_index == o._index)
            helper::copy_assign_same(_index, &_data, &o._data);
        else {
            helper::destruct(_index, &_data);
            helper::copy_construct(o._index, &_data, &o._data);
            _index = o._index;
        }
        return *this;
    }

    template <class T, class U = std::decay_t<T>,
              class = std::enable_if_t<contains<U, Types...>::value>>
    variant &operator=(T &&val) {
        if (is<U>())
            unsafe_as<U>() = std::forward<T>(val);
        else
            emplace<U>(std::forward<T>(val));
        return *this;
    }

    void swap(variant &o) {
        if (_index == o._index)
            helper::swap_same(_index, &_data, &o._data);
        else {
            variant tmp(*this);

            (*this) = o;
            o = tmp;
        }
    }

    template <class T, class... Args,
              class = std::enable_if<contains<T, Types...>::value>>
    void emplace(Args &&... args) {
        helper::destruct(_index, &_data);

        new (&_data) T(std::forward<Args>(args)...);
        _index = helper::template index_of<T>();
    }

    bool empty() const { return _index == 0; }

    unsigned index() const { return _index; }

    template <class T> static constexpr unsigned index_of() {
        return helper::template index_of<T>();
    }

    template <class T> bool is() const {
        return !empty() && _index == helper::template index_of<T>();
    }

    template <class T> T &as() {
        if (not is<T>())
            throw bad_variant_access{};
        return unsafe_as<T>();
    }

    template <class T> const T &as() const {
        if (not is<T>())
            throw bad_variant_access{};
        return unsafe_as<T>();
    }

    template <class T> T &unsafe_as() { return *reinterpret_cast<T *>(&_data); }

    template <class T> const T &unsafe_as() const {
        return *reinterpret_cast<const T *>(&_data);
    }

private:
    using helper = _impl::variant<Types...>;

    unsigned _index;
    std::aligned_union_t<1, Types...> _data;
};

template <class... T> void swap(variant<T...> &a, variant<T...> &b) {
    a.swap(b);
}

} // namespace ea
