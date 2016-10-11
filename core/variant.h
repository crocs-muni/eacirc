#pragma once

#include "traits.h"
#include <exception>
#include <utility>

namespace _impl {

    template <typename...> struct variant;

    template <> struct variant<> {
        static constexpr unsigned index = 0;
        template <typename Q> static constexpr unsigned index_of() { return 0; }

        static void destruct(const unsigned, void*) {}
        static void move_construct(const unsigned, void*, void*) {}
        static void copy_construct(const unsigned, void*, const void*) {}

        static void move_assign_same(const unsigned, void*, void*) {}
        static void copy_assign_same(const unsigned, void*, const void*) {}

        static void swap_same(const unsigned, void*, void*) {}
    };

    template <typename T, typename... Ts> struct variant<T, Ts...> {
        static constexpr unsigned index = 1 + variant<Ts...>::index;

        template <typename Q> static constexpr unsigned index_of() {
            return std::is_same<Q, T>::value ? index : variant<Ts...>::template index_of<Q>();
        }

        static void destruct(const unsigned i, void* data) {
            if (i == index)
                ref(data).~T();
            else
                variant<Ts...>::destruct(i, data);
        }

        static void move_construct(unsigned i, void* data, void* other) {
            if (i == index)
                new (data) T(move(other));
            else
                variant<Ts...>::move_construct(i, data, other);
        }

        static void copy_construct(const unsigned i, void* data, const void* other) {
            if (i == index)
                new (data) T(ref(other));
            else
                variant<Ts...>::copy_construct(i, data, other);
        }

        static void move_assign_same(const unsigned i, void* lhs, void* rhs) {
            if (i == index)
                ref(lhs) = move(rhs);
            else
                variant<Ts...>::move_assign_same(i, lhs, rhs);
        }

        static void copy_assign_same(const unsigned i, void* lhs, const void* rhs) {
            if (i == index)
                ref(lhs) = ref(rhs);
            else
                variant<Ts...>::copy_assign_same(i, lhs, rhs);
        }

        static void swap_same(const unsigned i, void* lhs, void* rhs) {
            using std::swap;

            if (i == index)
                swap(ref(lhs), ref(rhs));
            else
                variant<Ts...>::swap_same(i, lhs, rhs);
        }

        static T&& move(void* data) { return std::move(ref(data)); }

        static T& ref(void* data) { return *reinterpret_cast<T*>(data); }
        static T const& ref(const void* data) { return *reinterpret_cast<const T*>(data); }
    };

} // namespace _impl

struct bad_variant_access : std::exception {};

template <typename... Types> struct variant {
    static_assert(all_unique<Types...>::value, "Every type in variadic template must be unique.");

    variant()
        : _index(_impl::variant<>::index) {}

    variant(variant&& o)
        : _index(o._index) {
        helper::move_construct(_index, &_data, &o._data);
    }

    variant(const variant& o)
        : _index(o._index) {
        helper::copy_construct(_index, &_data, &o._data);
    }

    template <typename T,
              typename U = typename std::decay<T>::type,
              typename = typename std::enable_if<contains<U, Types...>::value>::type>
    variant(T&& val)
        : _index(helper::template index_of<U>()) {
        new (&_data) U(std::forward<T>(val));
    }

    ~variant() { helper::destruct(_index, &_data); }

    variant& operator=(variant&& o) {
        if (_index == o._index)
            helper::move_assign_same(_index, &_data, &o._data);
        else {
            helper::destruct(_index, &_data);
            helper::move_construct(o._index, &_data, &o._data);
            _index = o._index;
        }
        return *this;
    }

    variant& operator=(const variant& o) {
        if (_index == o._index)
            helper::copy_assign_same(_index, &_data, &o._data);
        else {
            helper::destruct(_index, &_data);
            helper::copy_construct(o._index, &_data, &o._data);
            _index = o._index;
        }
        return *this;
    }

    template <typename T,
              typename U = typename std::decay<T>::type,
              typename = typename std::enable_if<contains<U, Types...>::value>::type>
    variant& operator=(T&& val) {
        if (is<U>())
            unsafe_as<U>() = std::forward<T>(val);
        else
            emplace<U>(std::forward<T>(val));
        return *this;
    }

    void swap(variant& o) {
        if (_index == o._index)
            helper::swap_same(_index, &_data, &o._data);
        else {
            variant tmp(*this);

            (*this) = o;
            o = tmp;
        }
    }

    friend void swap(variant& a, variant& b) { a.swap(b); }

    template <typename T, typename... Args, class = std::enable_if<contains<T, Types...>::value>>
    void emplace(Args&&... args) {
        helper::destruct(_index, &_data);

        new (&_data) T(std::forward<Args>(args)...);
        _index = helper::template index_of<T>();
    }

    bool empty() const { return _index == 0; }

    unsigned index() const { return _index; }

    template <typename T> static constexpr unsigned index_of() {
        return helper::template index_of<T>();
    }

    template <typename T> bool is() const {
        return !empty() && _index == helper::template index_of<T>();
    }

    template <typename T> T& as() {
        if (not is<T>())
            throw bad_variant_access{};
        return unsafe_as<T>();
    }

    template <typename T> const T& as() const {
        if (not is<T>())
            throw bad_variant_access{};
        return unsafe_as<T>();
    }

    template <typename T> T& unsafe_as() { return *reinterpret_cast<T*>(&_data); }
    template <typename T> T const& unsafe_as() const { return *reinterpret_cast<const T*>(&_data); }

private:
    using helper = _impl::variant<Types...>;

    unsigned _index;
    typename std::aligned_union<1, Types...>::type _data;
};
