#include "settings.h"
#include <core/exceptions.h>
#include <core/logger.h>
#include <yaml-cpp/yaml.h>

using namespace settings;

namespace YAML {
template <> struct convert<Main> {
    static bool decode(Node const& node, Main& main) {
        main.enable_cuda = node["enable_cuda"].as<bool>();
        main.project = node["project"].as<std::string>();
        main.evaluator = node["evaluator"].as<std::string>();
        main.evaluator_precision = node["evaluator_precision"].as<unsigned>();
        main.significance_level = node["significance_level"].as<unsigned>();

        return true;
    }
};

template <> struct convert<Random> {
    static bool decode(Node const& node, Random& random) {
        random.use_fixed_seed = node["use_fixed_seed"].as<bool>();
        random.main_seed = node["seed"].as<u32>();
        random.qrng_path = node["qrng_path"].as<std::string>();

        return true;
    }
};

template <> struct convert<TestVectors> {
    static bool decode(Node const& node, TestVectors& tv) {
        tv.input_size = node["input_size"].as<unsigned>();
        tv.output_size = node["output_size"].as<unsigned>();
        tv.size_of_set = node["size_of_set"].as<unsigned>();
        tv.change_frequency = node["change_frequency"].as<unsigned>();

        return true;
    }
};

template <> struct convert<Circuit::FunctionSet> {
    static bool decode(Node const& node, Circuit::FunctionSet& set) {
        set.NOP = node["NOP"].as<bool>();
        set.CONS = node["CONS"].as<bool>();
        set.AND = node["AND"].as<bool>();
        set.NAND = node["NAND"].as<bool>();
        set.OR = node["OR"].as<bool>();
        set.XOR = node["XOR"].as<bool>();
        set.NOR = node["NOR"].as<bool>();
        set.NOT = node["NOT"].as<bool>();
        set.SHIL = node["SHIL"].as<bool>();
        set.SHIR = node["SHIR"].as<bool>();
        set.ROTL = node["ROTL"].as<bool>();
        set.ROTR = node["ROTR"].as<bool>();
        set.MASK = node["MASK"].as<bool>();

        return true;
    }
};

template <> struct convert<Circuit> {
    static bool decode(Node const& node, Circuit& circ) {
        circ.size_of_layer = node["size_of_layer"].as<unsigned>();
        circ.num_of_layers = node["num_of_layers"].as<unsigned>();
        circ.function_set = node["function_set"].as<Circuit::FunctionSet>();

        return true;
    }
};
} // namespace YAML

void load_settings(const std::string file) {
    try {
        auto config = YAML::LoadFile(file);

        auto main = config["main_settings"].as<Main>();
        auto random = config["random"].as<Random>();
        auto test_vectors = config["test_vectors"].as<TestVectors>();

    } catch (std::exception& e) {
        Logger::error() << e.what() << std::endl;
        throw fatal_error();
    }
}
