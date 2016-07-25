#pragma once

#include "ea-variant.h"
#include <memory>
#include <unordered_map>

namespace ea {

struct settings {
    settings() = default;
    settings(settings &&) = default;
    settings(const settings &) = default;

    settings &operator=(settings o) {
        swap(o);
        return *this;
    }

    void swap(settings &o) {
        using std::swap;
        swap(_value, o._value);
    }

    template <class T> T &as() { return _value.as<T>(); }
    template <class T> const T &as() const { return _value.as<T>(); }

    settings &operator[](std::string key) {
        if (!_value.is<dictionary>())
            _value.emplace<dictionary>();

        auto dict = _value.unsafe_as<dictionary>();
        auto proxy = dict[key];
        return *proxy;
    }

    const settings &operator[](std::string key) const {
        auto dict = _value.as<dictionary>();
        return *(dict.at(key));
    }

    bool empty() const noexcept { return _value.empty(); }

private:
    struct proxy {
        proxy()
            : _ptr(std::make_unique<settings>()) {}

        proxy(const proxy &o)
            : _ptr(std::make_unique<settings>(*o._ptr)) {}

        proxy &operator=(const proxy &o) {
            *_ptr = *o._ptr;
            return *this;
        }

        settings &operator*() { return *_ptr; }
        const settings &operator*() const { return *_ptr; }

    private:
        std::unique_ptr<settings> _ptr;
    };

    using dictionary = std::unordered_map<std::string, proxy>;

    variant<bool, double, std::uint64_t, std::string, dictionary> _value;
};

void swap(settings &a, settings &b) { a.swap(b); }

} // namespace ea
