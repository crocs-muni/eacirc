#pragma once

#include "exceptions.h"
#include "logger.h"
#include <functional>
#include <memory>
#include <unordered_map>

template <class Base> class Factory {
    std::unordered_map<std::string, std::function<std::unique_ptr<Base>()>> _handlers;

public:
    template <class Fn> void add(const std::string key, Fn&& handler) {
        bool res = false;
        std::tie(std::ignore, res) = _handlers.emplace(std::move(key), std::forward<Fn>(handler));

        if (!res) {
            Logger::error() << "Unable to register handler for [" << key << "]" << std::endl;
            throw fatal_error();
        }
    }

    std::unique_ptr<Base> create(const std::string key) {
        auto search = _handlers.find(key);

        if (search == _handlers.end()) {
            Logger::error() << "Unable to find handler for [" << key << "]" << std::endl;
            throw fatal_error();
        }

        return search->second();
    }
};
