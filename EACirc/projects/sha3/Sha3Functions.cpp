#include "Sha3Functions.h"

const char* Sha3Functions::sha3ToString(int algorithm) {
    switch(algorithm) {
    case SHA3_ABACUS:           return "Abacus";
    case SHA3_ARIRANG:          return "ARIRANG";
    case SHA3_AURORA:           return "Aurora";
    case SHA3_BLAKE:            return "Blake";
    case SHA3_BLENDER:          return "Blender";
    case SHA3_BMW:              return "BMW";
    case SHA3_BOOLE:            return "Boole";
    case SHA3_CHEETAH:          return "Cheetah";
    case SHA3_CHI:              return "CHI";
    case SHA3_CRUNCH:           return "CRUNCH";
    case SHA3_CUBEHASH:         return "CubeHash";
    case SHA3_DCH:              return "DCH";
    case SHA3_DYNAMICSHA:       return "DynamicSHA";
    case SHA3_DYNAMICSHA2:      return "DynamicSHA2";
    case SHA3_ECHO:             return "ECHO";
    case SHA3_ECOH:             return "ECOH";
    case SHA3_EDON:             return "EDON";
    case SHA3_ENRUPT:           return "EnRUPT";
    case SHA3_ESSENCE:          return "ESSENCE";
    case SHA3_FUGUE:            return "Fugue";
    case SHA3_GROSTL:           return "Grostl";
    case SHA3_HAMSI:            return "Hamsi";
    case SHA3_JH:               return "JH";
    case SHA3_KECCAK:           return "Keccak";
    case SHA3_KHICHIDI:         return "Khichidi";
    case SHA3_LANE:             return "Lane";
    case SHA3_LESAMNTA:         return "Lesamnta";
    case SHA3_LUFFA:            return "Luffa";
    case SHA3_LUX:              return "LUX";
    case SHA3_MSCSHA3:          return "MCSSHA3";
    case SHA3_MD6:              return "MD6";
    case SHA3_MESHHASH:         return "MeshHash";
    case SHA3_NASHA:            return "NaSHA";
    case SHA3_SANDSTORM:        return "SANDstorm";
    case SHA3_SARMAL:           return "Sarmal";
    case SHA3_SHABAL:           return "Shabal";
    case SHA3_SHAMATA:          return "Shameta";
    case SHA3_SHAVITE3:         return "SHAvite3";
    case SHA3_SIMD:             return "SIMD";
    case SHA3_SKEIN:            return "Skein";
    case SHA3_SPECTRALHASH:     return "SpectralHash";
    case SHA3_STREAMHASH:       return "StreamHash";
    case SHA3_SWIFFTX:          return "SWIFFTX";
    case SHA3_TANGLE:           return "Tangle";
    case SHA3_TIB3:             return "TIB3";
    case SHA3_TWISTER:          return "Twister";
    case SHA3_VORTEX:           return "Vortex";
    case SHA3_WAMM:             return "WaMM";
    case SHA3_WATERFALL:        return "Waterfall";
    case SHA3_RANDOM:           return "random data";
    default:                    return "(unknown hash function)";
    }
}
