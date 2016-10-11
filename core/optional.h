#pragma once

#include <stdexcept>
#include <type_traits>
#include <utility>

struct nullopt_t {
    constexpr explicit nullopt_t() = default;
} constexpr nullopt{};

struct in_place_t {
    constexpr explicit in_place_t() = default;
} constexpr in_place{};

struct bad_optional_access : std::logic_error {
    bad_optional_access()
        : logic_error("Optional does not contain value!") {}
};

template <typename T> struct optional {
    using value_type = T;

    constexpr optional() noexcept : optional(nullopt) {}
    constexpr optional(nullopt_t) noexcept : _set(false) {}

    optional(optional&& other)
        : _set(other._set) {
        if (_set)
            _construct(other._unsafe_move());
    }

    optional(const optional& other)
        : _set(other._set) {
        if (_set)
            _construct(other._unsafe_ref());
    }

    optional(value_type&& value) noexcept : _set(true) { _construct(std::move(value)); }

    optional(const value_type& value) noexcept : _set(true) { _construct(value); }

    template <typename... Args>
    explicit optional(in_place_t, Args&&... args)
        : _set(true) {
        _construct(std::forward<Args>(args)...);
    }

    ~optional() { reset(); }

    optional& operator=(nullopt_t) {
        reset();
        return *this;
    }

    optional& operator=(optional&& other) {
        if (_set && other._set)
            _unsafe_ref() = other._unsafe_move();
        else if (_set)
            _destruct();
        else if (other._set)
            _construct(other._unsafe_move());
        return *this;
    }

    optional& operator=(const optional& other) {
        if (_set && other._set)
            _unsafe_ref() = other._unsafe_ref();
        else if (_set)
            _destruct();
        else if (other._set)
            _construct(other._unsafe_ref());
        return *this;
    }

    template <typename U> optional& operator=(U&& value) {
        if (_set)
            _unsafe_ref() = std::forward<U>(value);
        else
            _construct(std::forward<U>(value));
        return *this;
    }

    void swap(optional& other) {
        using std::swap;

        if (_set && other._set)
            swap(_unsafe_ref(), other._unsafe_ref());
        else if (_set) {
            other._construct(_unsafe_move());
            _destruct();
        } else if (other._set) {
            _construct(other._unsafe_move());
            other._destruct();
        }
    }

    static void swap(optional& lhs, optional& rhs) { lhs.swap(rhs); }

public:
    bool has_value() const { return _set; }

    void reset() {
        if (_set)
            _destruct();
    }

    template <typename... Args> void emplace(Args&&... args) {
        reset();
        _construct(std::forward<Args>(args)...);
    }

    template <typename U, typename... Args>
    auto emplace(std::initializer_list<U> ilist, Args&&... args) -> typename std::enable_if<
            std::is_constructible<T, std::initializer_list<U>&, Args&&...>::value>::type {
        reset();
        _construct(ilist, std::forward<Args>(args)...);
    }

    value_type& value() & {
        if (!_set)
            throw bad_optional_access{};
        return _unsafe_ref();
    }

    const value_type& value() const& {
        if (!_set)
            throw bad_optional_access{};
        return _unsafe_ref();
    }

    value_type&& value() && { std::move(value()); }
    const value_type&& value() const&& { std::move(value()); }

    template <typename U> value_type value_or(U&& default_value) const& {
        return has_value() ? value() : static_cast<value_type>(std::forward<U>(default_value));
    }

    template <typename U> value_type value_or(U&& default_value) && {
        return has_value() ? std::move(value())
                           : static_cast<value_type>(std::forward<U>(default_value));
    }

public:
    operator bool() const { return has_value(); }

    value_type* operator->() { return &value(); }
    const value_type* operator->() const { return &value(); }

    value_type& operator*() & { return value(); }
    const value_type& operator*() const& { return value(); }

    value_type&& operator*() && { return value(); }
    const value_type&& operator*() const&& { return value(); }

public:
    friend bool operator==(const optional& lhs, const optional& rhs) {
        if (bool(lhs) != bool(rhs))
            return false;
        if (bool(lhs) == false)
            return true;
        return *lhs == *rhs;
    }

    friend bool operator!=(const optional& lhs, const optional& rhs) {
        if (bool(lhs) != bool(rhs))
            return true;
        if (bool(lhs) == false)
            return false;
        return *lhs != *rhs;
    }

    friend bool operator==(const optional& lhs, nullopt_t) { return !lhs; }
    friend bool operator==(nullopt_t, const optional& rhs) { return !rhs; }

    friend bool operator!=(const optional& lhs, nullopt_t) { return bool(lhs); }
    friend bool operator!=(nullopt_t, const optional& rhs) { return bool(rhs); }

    // clang-format off
    friend bool operator==(const optional& lhs, const value_type& rhs) { return bool(lhs) ? *lhs == rhs : false; }
    friend bool operator==(const value_type& lhs, const optional& rhs) { return bool(rhs) ? lhs == *rhs : false; }

    friend bool operator!=(const optional& lhs, const value_type& rhs) { return bool(lhs) ? *lhs != rhs : true; }
    friend bool operator!=(const value_type& lhs, const optional& rhs) { return bool(rhs) ? lhs != *rhs : true; }
    // clang-format on

private:
    bool _set;
    typename std::aligned_storage<sizeof(value_type), alignof(value_type)>::type _storage;

    value_type& _unsafe_ref() { return reinterpret_cast<value_type&>(_storage); }
    const value_type& _unsafe_ref() const { return reinterpret_cast<const value_type&>(_storage); }

    value_type&& _unsafe_move() { return std::move(_unsafe_ref()); }
    const value_type&& _unsafe_move() const { return std::move(_unsafe_ref()); }

    template <typename... Args> void _construct(Args&&... args) {
        new (&_storage) value_type(std::forward<Args>(args)...);
        _set = true;
    }

    void _destruct() {
        _unsafe_ref().~value_type();
        _set = false;
    }
};
