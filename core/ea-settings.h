#pragma once

#include "ea-variant.h"
#include <memory>
#include <unordered_map>

namespace ea {

struct settings {
    void swap(settings &o) {
        using std::swap;
        swap(_value, o._value);
    }

    settings &operator=(bool val) {
        _value = val;
        return *this;
    }

    settings &operator=(long long val) {
        _value = val;
        return *this;
    }

    settings &operator=(double val) {
        _value = val;
        return *this;
    }

    settings &operator=(std::string val) {
        _value = std::move(val);
        return *this;
    }

    operator const bool() const { return _value.as<bool>(); }
    operator const long long() const { return _value.as<long long>(); }
    operator const double() const { return _value.as<double>(); }
    operator const std::string &() const { return _value.as<std::string>(); }

    settings &operator[](const char *key) {
        if (!_value.is<dictionary>())
            _value.emplace<dictionary>();

        auto dict = _value.unsafe_as<dictionary>();
        auto proxy = dict[key];
        return *proxy;
    }

    const settings &operator[](const char *key) const {
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

    variant<bool, long long, double, std::string, dictionary> _value;
};

void swap(settings &a, settings &b) { a.swap(b); }

} // namespace ea
