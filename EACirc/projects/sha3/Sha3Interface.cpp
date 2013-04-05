#include "Sha3Interface.h"
#include "EACglobals.h"
#include "hash_functions/hashFunctions.h"

const char* Sha3Interface::sha3ToString(int algorithm) {
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

Sha3Interface* Sha3Interface::getSha3Function(int algorithm, int numRounds) {
    switch (algorithm) {
    case SHA3_ABACUS: { return new Abacus(numRounds); break; }
    case SHA3_ARIRANG: { return new Arirang(numRounds); break; }
    case SHA3_AURORA: { return new Aurora(numRounds); break; }
    case SHA3_BLAKE: { return new Blake(numRounds); break; }
    case SHA3_BLENDER: { return new Blender(numRounds); break; }
    case SHA3_BMW: { return new BMW(numRounds); break; }
    case SHA3_BOOLE: { return new Boole(numRounds); break; }
    case SHA3_CHEETAH: { return new Cheetah(numRounds); break; }
    case SHA3_CHI: { return new Chi(numRounds); break; }
    case SHA3_CRUNCH: { return new Crunch(numRounds); break; }
    case SHA3_CUBEHASH: { return new Cubehash(numRounds); break; }
    case SHA3_DCH: { return new DCH(numRounds); break; }
    case SHA3_DYNAMICSHA: { return new DSHA(numRounds); break; }
    case SHA3_DYNAMICSHA2: { return new DSHA2(numRounds); break; }
    case SHA3_ECHO: { return new Echo(numRounds); break; }
    // case SHA3_ECOH: { return new Ecoh(); break; }
    case SHA3_EDON: { return new Edon(); break; }
    case SHA3_ESSENCE: { return new Essence(numRounds); break; }
    // case SHA3_ENRUPT: { return new Enrupt(); break; }
    case SHA3_FUGUE: { return new Fugue(numRounds); break; }
    case SHA3_GROSTL: { return new Grostl(numRounds); break; }
    case SHA3_HAMSI: { return new Hamsi(numRounds); break; }
    case SHA3_JH: { return new JH(numRounds); break; }
    case SHA3_KECCAK: { return new Keccak(); break; }
    case SHA3_KHICHIDI: { return new Khichidi(); break; }
    case SHA3_LANE: { return new Lane(numRounds); break; }
    case SHA3_LESAMNTA: { return new Lesamnta(numRounds); break; }
    case SHA3_LUFFA: { return new Luffa(numRounds); break; }
    // case SHA3_LUX: { return new Lux(numRounds); break; }
    case SHA3_MSCSHA3: { return new Mscsha(); break; }
    case SHA3_MD6: { return new MD6(numRounds); break; }
    case SHA3_MESHHASH: { return new MeshHash(numRounds); break; }
    case SHA3_NASHA: { return new Nasha(); break; }
    // case SHA3_SANDSTORM: { return new SandStorm(numRounds); break; }
    case SHA3_SARMAL: { return new Sarmal(numRounds); break; }
    case SHA3_SHABAL: { return new Shabal(); break; }
    case SHA3_SHAMATA: { return new Shamata(); break; }
    case SHA3_SHAVITE3: { return new SHAvite(numRounds); break; }
    case SHA3_SIMD: { return new Simd(numRounds); break; }
    case SHA3_SKEIN: { return new Skein(); break; }
    case SHA3_SPECTRALHASH: { return new SpectralHash(); break; }
    case SHA3_STREAMHASH: { return new StreamHash(); break; }
    // case SHA3_SWIFFTX: { return new Swifftx(); break; }
    case SHA3_TANGLE: { return new Tangle(numRounds); break; }
    // case SHA3_TIB3: { return new Tib(numRounds); break; }
    case SHA3_TWISTER: { return new Twister(numRounds); break; }
    case SHA3_VORTEX: { return new Vortex(numRounds); break; }
    case SHA3_WAMM: { return new WaMM(numRounds); break; }
    case SHA3_WATERFALL: { return new Waterfall(numRounds); break; }
    default:
        mainLogger.out(LOGGER_ERROR) << "Unknown hash function type (" << algorithm << ")." << endl;
        return NULL;
    }
}
