#pragma once

#include "variant.h"
#include "view.h"
#include <iomanip>
#include <unordered_map>
#include <vector>

template <typename Config> struct cmd {
  struct proxy {
    template <class T> using pointer = T Config::*;

    using value_type = core::variant<pointer<bool>, pointer<std::string>>;

    template <class T>
    proxy(pointer<T> ptr)
        : _ptr(ptr) {}

    void parse(Config& cfg, std::string val) const {
      switch (_ptr.index()) {
      case value_type::template index_of<pointer<bool>>(): {
        auto ptr = _ptr.template as<pointer<bool>>();
        cfg.*ptr = true;
        break;
      }
      case value_type::template index_of<pointer<std::string>>(): {
        auto ptr = _ptr.template as<pointer<std::string>>();
        cfg.*ptr = val;
        break;
      }
      }
    }

  private:
    value_type _ptr;
  };

  struct opt {
    const std::string flag;
    const std::string name;
    const std::string desc;
    const proxy value;
  };

  cmd(std::initializer_list<opt> opts)
      : _opts(opts) {
    for (const auto& opt : _opts) {
      if (!_mapping.emplace(opt.flag, opt.value).second)
        throw std::invalid_argument("an option \"" + opt.flag + "\" is already defined.");

      if (!_mapping.emplace(opt.name, opt.value).second)
        throw std::invalid_argument("an option \"" + opt.name + "\" is already defined.");
    }
  }

  Config parse(view<const char**> args) {
    Config config{};

    for (const std::string arg : args.drop(1)) {
      auto n = arg.find('=');
      auto fst = arg.substr(0, n);
      auto snd = (n != std::string::npos) ? arg.substr(n + 1) : "";

      auto it = _mapping.find(fst);

      if (it == _mapping.end())
        throw std::runtime_error("an unknown cmd option \"" + fst + "\"");
      if (arg.back() == '=')
        throw std::runtime_error("an empty value for an option \"" + fst + "\" is not allowed");

      it->second.parse(config, snd);
    }
    return config;
  }

  void print(std::ostream& out) const {
    out << "Options:" << std::endl;
    for (auto& opt : _opts) {
      std::string name = "  " + opt.flag + ", " + opt.name + " ";

      out << std::left << std::setw(28) << name << opt.desc << std::endl;
    }
  }

private:
  std::vector<opt> _opts;
  std::unordered_map<std::string, proxy> _mapping;
};
