#include "factory.h"
#include "hash_functions/hash_functions.h"
#include "sha3_interface.h"

// clang-format off
namespace sha3 {

    std::string to_string(sha3_algorithm alg) {
        switch (alg) {
        case sha3_algorithm::ABACUS:           return "Abacus";
        case sha3_algorithm::ARIRANG:          return "ARIRANG";
        case sha3_algorithm::AURORA:           return "Aurora";
        case sha3_algorithm::BLAKE:            return "Blake";
        case sha3_algorithm::BLENDER:          return "Blender";
        case sha3_algorithm::BMW:              return "BMW";
        case sha3_algorithm::BOOLE:            return "Boole";
        case sha3_algorithm::CHEETAH:          return "Cheetah";
        case sha3_algorithm::CHI:              return "CHI";
        case sha3_algorithm::CRUNCH:           return "CRUNCH";
        case sha3_algorithm::CUBEHASH:         return "CubeHash";
        case sha3_algorithm::DCH:              return "DCH";
        case sha3_algorithm::DYNAMICSHA:       return "DynamicSHA";
        case sha3_algorithm::DYNAMICSHA2:      return "DynamicSHA2";
        case sha3_algorithm::ECHO:             return "ECHO";
        case sha3_algorithm::ECOH:             return "ECOH";
        case sha3_algorithm::EDON:             return "EDON";
        case sha3_algorithm::ENRUPT:           return "EnRUPT";
        case sha3_algorithm::ESSENCE:          return "ESSENCE";
        case sha3_algorithm::FUGUE:            return "Fugue";
        case sha3_algorithm::GROSTL:           return "Grostl";
        case sha3_algorithm::HAMSI:            return "Hamsi";
        case sha3_algorithm::JH:               return "JH";
        case sha3_algorithm::KECCAK:           return "Keccak";
        case sha3_algorithm::KHICHIDI:         return "Khichidi";
        case sha3_algorithm::LANE:             return "Lane";
        case sha3_algorithm::LESAMNTA:         return "Lesamnta";
        case sha3_algorithm::LUFFA:            return "Luffa";
        case sha3_algorithm::LUX:              return "LUX";
        case sha3_algorithm::MSCSHA3:          return "MCSSHA3";
        case sha3_algorithm::MD6:              return "MD6";
        case sha3_algorithm::MESHHASH:         return "MeshHash";
        case sha3_algorithm::NASHA:            return "NaSHA";
        case sha3_algorithm::SANDSTORM:        return "SANDstorm";
        case sha3_algorithm::SARMAL:           return "Sarmal";
        case sha3_algorithm::SHABAL:           return "Shabal";
        case sha3_algorithm::SHAMATA:          return "Shameta";
        case sha3_algorithm::SHAVITE3:         return "SHAvite3";
        case sha3_algorithm::SIMD:             return "SIMD";
        case sha3_algorithm::SKEIN:            return "Skein";
        case sha3_algorithm::SPECTRALHASH:     return "SpectralHash";
        case sha3_algorithm::STREAMHASH:       return "StreamHash";
        case sha3_algorithm::SWIFFTX:          return "SWIFFTX";
        case sha3_algorithm::TANGLE:           return "Tangle";
        case sha3_algorithm::TIB3:             return "TIB3";
        case sha3_algorithm::TWISTER:          return "Twister";
        case sha3_algorithm::VORTEX:           return "Vortex";
        case sha3_algorithm::WAMM:             return "WaMM";
        case sha3_algorithm::WATERFALL:        return "Waterfall";
        case sha3_algorithm::TANGLE2:          return "Tangle2";
        }
    }

    sha3_algorithm algorithm_from_string(std::string str) {
        if (str == to_string(sha3_algorithm::ABACUS))           return sha3_algorithm::ABACUS;
        if (str == to_string(sha3_algorithm::ARIRANG))          return sha3_algorithm::ARIRANG;
        if (str == to_string(sha3_algorithm::AURORA))           return sha3_algorithm::AURORA;
        if (str == to_string(sha3_algorithm::BLAKE))            return sha3_algorithm::BLAKE;
        if (str == to_string(sha3_algorithm::BLENDER))          return sha3_algorithm::BLENDER;
        if (str == to_string(sha3_algorithm::BMW))              return sha3_algorithm::BMW;
        if (str == to_string(sha3_algorithm::BOOLE))            return sha3_algorithm::BOOLE;
        if (str == to_string(sha3_algorithm::CHEETAH))          return sha3_algorithm::CHEETAH;
        if (str == to_string(sha3_algorithm::CHI))              return sha3_algorithm::CHI;
        if (str == to_string(sha3_algorithm::CRUNCH))           return sha3_algorithm::CRUNCH;
        if (str == to_string(sha3_algorithm::CUBEHASH))         return sha3_algorithm::CUBEHASH;
        if (str == to_string(sha3_algorithm::DCH))              return sha3_algorithm::DCH;
        if (str == to_string(sha3_algorithm::DYNAMICSHA))       return sha3_algorithm::DYNAMICSHA;
        if (str == to_string(sha3_algorithm::DYNAMICSHA2))      return sha3_algorithm::DYNAMICSHA2;
        if (str == to_string(sha3_algorithm::ECHO))             return sha3_algorithm::ECHO;
        if (str == to_string(sha3_algorithm::ECOH))             return sha3_algorithm::ECOH;
        if (str == to_string(sha3_algorithm::EDON))             return sha3_algorithm::EDON;
        if (str == to_string(sha3_algorithm::ENRUPT))           return sha3_algorithm::ENRUPT;
        if (str == to_string(sha3_algorithm::ESSENCE))          return sha3_algorithm::ESSENCE;
        if (str == to_string(sha3_algorithm::FUGUE))            return sha3_algorithm::FUGUE;
        if (str == to_string(sha3_algorithm::GROSTL))           return sha3_algorithm::GROSTL;
        if (str == to_string(sha3_algorithm::HAMSI))            return sha3_algorithm::HAMSI;
        if (str == to_string(sha3_algorithm::JH))               return sha3_algorithm::JH;
        if (str == to_string(sha3_algorithm::KECCAK))           return sha3_algorithm::KECCAK;
        if (str == to_string(sha3_algorithm::KHICHIDI))         return sha3_algorithm::KHICHIDI;
        if (str == to_string(sha3_algorithm::LANE))             return sha3_algorithm::LANE;
        if (str == to_string(sha3_algorithm::LESAMNTA))         return sha3_algorithm::LESAMNTA;
        if (str == to_string(sha3_algorithm::LUFFA))            return sha3_algorithm::LUFFA;
        if (str == to_string(sha3_algorithm::LUX))              return sha3_algorithm::LUX;
        if (str == to_string(sha3_algorithm::MSCSHA3))          return sha3_algorithm::MSCSHA3;
        if (str == to_string(sha3_algorithm::MD6))              return sha3_algorithm::MD6;
        if (str == to_string(sha3_algorithm::MESHHASH))         return sha3_algorithm::MESHHASH;
        if (str == to_string(sha3_algorithm::NASHA))            return sha3_algorithm::NASHA;
        if (str == to_string(sha3_algorithm::SANDSTORM))        return sha3_algorithm::SANDSTORM;
        if (str == to_string(sha3_algorithm::SARMAL))           return sha3_algorithm::SARMAL;
        if (str == to_string(sha3_algorithm::SHABAL))           return sha3_algorithm::SHABAL;
        if (str == to_string(sha3_algorithm::SHAMATA))          return sha3_algorithm::SHAMATA;
        if (str == to_string(sha3_algorithm::SHAVITE3))         return sha3_algorithm::SHAVITE3;
        if (str == to_string(sha3_algorithm::SIMD))             return sha3_algorithm::SIMD;
        if (str == to_string(sha3_algorithm::SKEIN))            return sha3_algorithm::SKEIN;
        if (str == to_string(sha3_algorithm::SPECTRALHASH))     return sha3_algorithm::SPECTRALHASH;
        if (str == to_string(sha3_algorithm::STREAMHASH))       return sha3_algorithm::STREAMHASH;
        if (str == to_string(sha3_algorithm::SWIFFTX))          return sha3_algorithm::SWIFFTX;
        if (str == to_string(sha3_algorithm::TANGLE))           return sha3_algorithm::TANGLE;
        if (str == to_string(sha3_algorithm::TIB3))             return sha3_algorithm::TIB3;
        if (str == to_string(sha3_algorithm::TWISTER))          return sha3_algorithm::TWISTER;
        if (str == to_string(sha3_algorithm::VORTEX))           return sha3_algorithm::VORTEX;
        if (str == to_string(sha3_algorithm::WAMM))             return sha3_algorithm::WAMM;
        if (str == to_string(sha3_algorithm::WATERFALL))        return sha3_algorithm::WATERFALL;
        if (str == to_string(sha3_algorithm::TANGLE2))          return sha3_algorithm::TANGLE2;
        throw std::invalid_argument("such SHA-3 algorithm named \"" + str + "\" does not exists");
    }

    static void _check_rounds(sha3_algorithm alg, unsigned rounds) {
        if (rounds > 0)
            throw std::logic_error{"an algorithm " + to_string(alg) + " cannot be limited in rounds!"};
    }

    std::unique_ptr<sha3_interface> create_algorithm(sha3_algorithm algorithm, unsigned rounds) {
        switch (algorithm) {
        case sha3_algorithm::ABACUS:       return std::make_unique<Abacus>(rounds);
        case sha3_algorithm::ARIRANG:      return std::make_unique<Arirang>(rounds);
        case sha3_algorithm::AURORA:       return std::make_unique<Aurora>(rounds);
        case sha3_algorithm::BLAKE:        return std::make_unique<Blake>(rounds);
        case sha3_algorithm::BLENDER:      return std::make_unique<Blender>(rounds);
        case sha3_algorithm::BMW:          return std::make_unique<BMW>(rounds);
        case sha3_algorithm::BOOLE:        return std::make_unique<Boole>(rounds);
        case sha3_algorithm::CHEETAH:      return std::make_unique<Cheetah>(rounds);
        case sha3_algorithm::CHI:          return std::make_unique<Chi>(rounds);
        case sha3_algorithm::CRUNCH:       return std::make_unique<Crunch>(rounds);
        case sha3_algorithm::CUBEHASH:     return std::make_unique<Cubehash>(rounds);
        case sha3_algorithm::DCH:          return std::make_unique<DCH>(rounds);
        case sha3_algorithm::DYNAMICSHA:   return std::make_unique<DSHA>(rounds);
        case sha3_algorithm::DYNAMICSHA2:  return std::make_unique<DSHA2>(rounds);
        case sha3_algorithm::ECHO:         return std::make_unique<Echo>(rounds);
        case sha3_algorithm::ECOH:        // return std::make_unique<Ecoh>();
            throw std::logic_error("requested algorithm \"" + to_string(algorithm) + "\" is not functional");
        case sha3_algorithm::EDON:         _check_rounds(algorithm, rounds);
                                           return std::make_unique<Edon>();
        case sha3_algorithm::ENRUPT:      // return std::make_unique<Enrupt>();
            throw std::logic_error("requested algorithm \"" + to_string(algorithm) + "\" is not functional");
        case sha3_algorithm::ESSENCE:      return std::make_unique<Essence>(rounds);
        case sha3_algorithm::FUGUE:        return std::make_unique<Fugue>(rounds);
        case sha3_algorithm::GROSTL:       return std::make_unique<Grostl>(rounds);
        case sha3_algorithm::HAMSI:        return std::make_unique<Hamsi>(rounds);
        case sha3_algorithm::JH:           return std::make_unique<JH>(rounds);
        case sha3_algorithm::KECCAK:       _check_rounds(algorithm, rounds);
                                           return std::make_unique<Keccak>();
        case sha3_algorithm::KHICHIDI:     _check_rounds(algorithm, rounds);
                                           return std::make_unique<Khichidi>();
        case sha3_algorithm::LANE:         return std::make_unique<Lane>(rounds);
        case sha3_algorithm::LESAMNTA:     return std::make_unique<Lesamnta>(rounds);
        case sha3_algorithm::LUFFA:        return std::make_unique<Luffa>(rounds);
        case sha3_algorithm::LUX:         // return std::make_unique<Lux>(rounds);
            throw std::logic_error("requested algorithm \"" + to_string(algorithm) + "\" is not functional");
        case sha3_algorithm::MSCSHA3:      _check_rounds(algorithm, rounds);
                                           return std::make_unique<Mscsha>();
        case sha3_algorithm::MD6:          return std::make_unique<MD6>(rounds);
        case sha3_algorithm::MESHHASH:     return std::make_unique<MeshHash>(rounds);
        case sha3_algorithm::NASHA:        _check_rounds(algorithm, rounds);
                                           return std::make_unique<Nasha>();
        case sha3_algorithm::SANDSTORM:   // return std::make_unique<SandStorm>(rounds);
            throw std::logic_error("requested algorithm \"" + to_string(algorithm) + "\" is not functional");
        case sha3_algorithm::SARMAL:       return std::make_unique<Sarmal>(rounds);
        case sha3_algorithm::SHABAL:       _check_rounds(algorithm, rounds);
                                           return std::make_unique<Shabal>();
        case sha3_algorithm::SHAMATA:      _check_rounds(algorithm, rounds);
                                           return std::make_unique<Shamata>();
        case sha3_algorithm::SHAVITE3:     return std::make_unique<SHAvite>(rounds);
        case sha3_algorithm::SIMD:         return std::make_unique<Simd>(rounds);
        case sha3_algorithm::SKEIN:        _check_rounds(algorithm, rounds);
                                           return std::make_unique<Skein>();
        case sha3_algorithm::SPECTRALHASH: _check_rounds(algorithm, rounds);
                                           return std::make_unique<SpectralHash>();
        case sha3_algorithm::STREAMHASH:   _check_rounds(algorithm, rounds);
                                           return std::make_unique<StreamHash>();
        case sha3_algorithm::SWIFFTX:     // return std::make_unique<Swifftx>();
            throw std::logic_error("requested algorithm \"" + to_string(algorithm) + "\" is not functional");
        case sha3_algorithm::TANGLE:       return std::make_unique<Tangle>(rounds);
        case sha3_algorithm::TIB3:        // return std::make_unique<Tib>(rounds);
            throw std::logic_error("requested algorithm \"" + to_string(algorithm) + "\" is not functional");
        case sha3_algorithm::TWISTER:      return std::make_unique<Twister>(rounds);
        case sha3_algorithm::VORTEX:       return std::make_unique<Vortex>(rounds);
        case sha3_algorithm::WAMM:         return std::make_unique<WaMM>(rounds);
        case sha3_algorithm::WATERFALL:    return std::make_unique<Waterfall>(rounds);
        case sha3_algorithm::TANGLE2:      return std::make_unique<Tangle2>(rounds);
        }
    }

} // namespace sha3
// clang-format on
