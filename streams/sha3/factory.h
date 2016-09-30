#pragma once

#include <memory>

struct sha3_interface;

namespace sha3 {

    enum class sha3_algorithm {
        ABACUS,
        ARIRANG,
        AURORA,
        BLAKE,
        BLENDER,
        BMW,
        BOOLE,
        CHEETAH,
        CHI,
        CRUNCH,
        CUBEHASH,
        DCH,
        DYNAMICSHA,
        DYNAMICSHA2,
        ECHO,
        ECOH,
        EDON,
        ENRUPT,
        ESSENCE,
        FUGUE,
        GROSTL,
        HAMSI,
        JH,
        KECCAK,
        KHICHIDI,
        LANE,
        LESAMNTA,
        LUFFA,
        LUX,
        MSCSHA3,
        MD6,
        MESHHASH,
        NASHA,
        SANDSTORM,
        SARMAL,
        SHABAL,
        SHAMATA,
        SHAVITE3,
        SIMD,
        SKEIN,
        SPECTRALHASH,
        STREAMHASH,
        SWIFFTX,
        TANGLE,
        TIB3,
        TWISTER,
        VORTEX,
        WAMM,
        WATERFALL,
        TANGLE2,
    };

    std::string to_string(sha3_algorithm algorithm);

    sha3_algorithm algorithm_from_string(std::string name);

    std::unique_ptr<sha3_interface> create_algorithm(sha3_algorithm algortihm, unsigned rounds);

} // namespace sha3
