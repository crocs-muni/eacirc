# === sha3 ===
cmake_minimum_required(VERSION 3.0.2)


include(${PROJECT_SOURCE_DIR}/cmake/eacirc_project.cmake)


set(SOURCES
    Sha3Project.cpp
    Hasher.cpp
    Sha3Interface.cpp
    Sha3Functions.cpp
# === SHA-3 hash function files ===
    hash_functions/Abacus/Abacus_sha3.cpp
    hash_functions/ARIRANG/Arirang_OP32.c
    hash_functions/ARIRANG/Arirang_sha3.cpp
    hash_functions/Aurora/Aurora_sha3.cpp
    hash_functions/Blake/Blake_sha3.cpp
    hash_functions/Blender/Blender_sha3.cpp
    hash_functions/BMW/BMW_sha3.cpp
    hash_functions/Boole/Boole_sha3.cpp
    hash_functions/Cheetah/Cheetah_sha3.cpp
    hash_functions/CHI/chi-fast32.c
    hash_functions/CHI/Chi_sha3.cpp
    hash_functions/CRUNCH/const_32.c
    hash_functions/CRUNCH/crunch_224.c
    hash_functions/CRUNCH/crunch_256.c
    hash_functions/CRUNCH/crunch_384.c
    hash_functions/CRUNCH/crunch_512.c
    hash_functions/CRUNCH/Crunch_sha3.cpp
    hash_functions/CubeHash/CubeHash_sha3.cpp
    hash_functions/DCH/DCH_sha3.cpp
    hash_functions/DynamicSHA2/DSHA2_sha3.cpp
    hash_functions/DynamicSHA/DSHA_sha3.cpp
    hash_functions/ECHO/Echo_sha3.cpp
    hash_functions/EDON/Edon_sha3.cpp
    hash_functions/ESSENCE/essence_compress_256.c
    hash_functions/ESSENCE/essence_compress_512.c
    hash_functions/ESSENCE/essence_L_tables.c
    hash_functions/ESSENCE/Essence_sha3.cpp
    hash_functions/Fugue/fugue_256.c
    hash_functions/Fugue/fugue_384.c
    hash_functions/Fugue/fugue_512.c
    hash_functions/Fugue/fugue.c
    hash_functions/Fugue/Fugue_sha3.cpp
    hash_functions/Grostl/Grostl_sha3.cpp
    hash_functions/Hamsi/hamsi-exp.c
    hash_functions/Hamsi/Hamsi_sha3.cpp
    hash_functions/Hamsi/hamsi-tables.c
    hash_functions/Hamsi/i.hamsi-ref.c
    hash_functions/JH/JH_sha3.cpp
    hash_functions/Keccak/KeccakDuplex.c
    hash_functions/Keccak/KeccakF-1600-opt32.c
    hash_functions/Keccak/Keccak_sha3.cpp
    hash_functions/Keccak/KeccakSponge.c
    hash_functions/Khichidi/khichidi_core.cpp
    hash_functions/Khichidi/Khichidi_sha3.cpp
    hash_functions/Lane/Lane_sha3.cpp
    hash_functions/Lesamnta/Lesamnta_sha3.cpp
    hash_functions/Luffa/Luffa_sha3.cpp
    hash_functions/MCSSHA3/Mcssha_sha3.cpp
    hash_functions/MD6/md6_compress.c
    hash_functions/MD6/md6_mode.c
    hash_functions/MD6/MD6_sha3.cpp
    hash_functions/MeshHash/MeshHash_sha3.cpp
    hash_functions/NaSHA/Nasha_sha3.cpp
    hash_functions/Sarmal/Sarmal_sha3.cpp
    hash_functions/Shabal/Shabal_sha3.cpp
    hash_functions/Shamata/Shamata_sha3.cpp
    hash_functions/SHAvite3/SHAvite_sha3.cpp
    hash_functions/SIMD/optimized.c
    hash_functions/SIMD/Simd_sha3.cpp
    hash_functions/Skein/skein_block.cpp
    hash_functions/Skein/skein.cpp
    hash_functions/Skein/skein_debug.cpp
    hash_functions/Skein/Skein_sha3.cpp
    hash_functions/SpectralHash/SpectralHash_sha3.cpp
    hash_functions/StreamHash/StreamHash_sha3.cpp
    hash_functions/Tangle/Tangle_sha3.cpp
    hash_functions/Twister/Twister_sha3.cpp
    hash_functions/Vortex/vortex_core.c
    hash_functions/Vortex/vortex_misc.c
    hash_functions/Vortex/Vortex_sha3.cpp
    hash_functions/Vortex/vortex_tables.c
    hash_functions/WaMM/BitArray.c
    hash_functions/WaMM/ReverseBits.c
    hash_functions/WaMM/WaMM.c
    hash_functions/WaMM/WaMMErrorMessage.c
    hash_functions/WaMM/WaMMOperator.c
    hash_functions/WaMM/Wamm_sha3.cpp
    hash_functions/WaMM/WaMMTappingPrimes.c
    hash_functions/Waterfall/Waterfall_sha3.cpp
    #    hash_functions/ECOH/ecoh.param.cpp
    #    hash_functions/ECOH/Ecoh_sha3.cpp
    #    hash_functions/EnRUPT/EnRUPT_opt.c
    #    hash_functions/EnRUPT/Enrupt_sha3.cpp
    #    hash_functions/LUX/Lux_sha3.cpp
    #    hash_functions/SANDstorm/Sandstorm_sha3.cpp
    #    hash_functions/SWIFFTX/SWIFFTX.c
    #    hash_functions/SWIFFTX/Swifftx_sha3.cpp
    #    hash_functions/TIB3/inupfin256.c
    #    hash_functions/TIB3/inupfin512.c
    #    hash_functions/TIB3/Tib_sha3.cpp
)

set(HEADERS
    Sha3Project.h
    Sha3Interface.h
    Sha3Constants.h
    Hasher.h
    Sha3Functions.h
# === SHA-3 hash function files ===
    hash_functions/hashFunctions.h
    hash_functions/Abacus/Abacus_sha3.h
    hash_functions/ARIRANG/Arirang_OP32.h
    hash_functions/ARIRANG/Arirang_sha3.h
    hash_functions/Aurora/Aurora_sha3.h
    hash_functions/Blake/Blake_sha3.h
    hash_functions/Blender/Blender_sha3.h
    hash_functions/BMW/BMW_sha3.h
    hash_functions/Boole/Boole_sha3.h
    hash_functions/Cheetah/Cheetah_sha3.h
    hash_functions/CHI/chi.h
    hash_functions/CHI/Chi_sha3.h
    hash_functions/CRUNCH/crunch_224.h
    hash_functions/CRUNCH/crunch_256.h
    hash_functions/CRUNCH/crunch_384.h
    hash_functions/CRUNCH/crunch_512.h
    hash_functions/CRUNCH/Crunch_sha3.h
    hash_functions/CRUNCH/crunch_type.h
    hash_functions/CubeHash/CubeHash_sha3.h
    hash_functions/DCH/DCH_sha3.h
    hash_functions/DynamicSHA2/DSHA2_sha3.h
    hash_functions/DynamicSHA/DSHA_sha3.h
    hash_functions/ECHO/Echo_sha3.h
    hash_functions/EDON/Edon_sha3.h
    hash_functions/ESSENCE/essence.h
    hash_functions/ESSENCE/Essence_sha3.h
    hash_functions/Fugue/aestab.h
    hash_functions/Fugue/aestab_t.h
    hash_functions/Fugue/fugue_256.h
    hash_functions/Fugue/fugue_384.h
    hash_functions/Fugue/fugue_512.h
    hash_functions/Fugue/fugue.h
    hash_functions/Fugue/Fugue_sha3.h
    hash_functions/Fugue/fugue_t.h
    hash_functions/Grostl/brg_endian.h
    hash_functions/Grostl/brg_types.h
    hash_functions/Grostl/Grostl_sha3.h
    hash_functions/Grostl/tables.h
    hash_functions/Hamsi/hamsi.h
    hash_functions/Hamsi/hamsi-internals.h
    hash_functions/Hamsi/Hamsi_sha3.h
    hash_functions/Hamsi/hamsi-tables.h
    hash_functions/JH/JH_sha3.h
    hash_functions/Keccak/KeccakSponge.h
    hash_functions/Keccak/brg_endian.h
    hash_functions/Keccak/KeccakDuplex.h
    hash_functions/Keccak/KeccakF-1600-interface.h
    hash_functions/Keccak/KeccakF-1600-int-set.h
    hash_functions/Keccak/KeccakF-1600-opt32-settings.h
    hash_functions/Keccak/Keccak_sha3.h
    hash_functions/Khichidi/common.h
    hash_functions/Khichidi/khichidi_core.h
    hash_functions/Khichidi/Khichidi_sha3.h
    hash_functions/Lane/Lane_sha3.h
    hash_functions/Lesamnta/Lesamnta_sha3.h
    hash_functions/Luffa/Luffa_sha3.h
    hash_functions/MCSSHA3/Mcssha_sha3.h
    hash_functions/MD6/inttypes.h
    hash_functions/MD6/md6.h
    hash_functions/MD6/MD6_sha3.h
    hash_functions/MD6/stdint.h
    hash_functions/MeshHash/MeshHash_sha3.h
    hash_functions/NaSHA/brg_endian.h
    hash_functions/NaSHA/brg_types.h
    hash_functions/NaSHA/Nasha_sha3.h
    hash_functions/Sarmal/Sarmal_sha3.h
    hash_functions/Shabal/Shabal_sha3.h
    hash_functions/Shamata/Shamata_sha3.h
    hash_functions/SHAvite3/AESround.h
    hash_functions/SHAvite3/portable.h
    hash_functions/SHAvite3/SHAvite3-256.h
    hash_functions/SHAvite3/SHAvite3-512.h
    hash_functions/SHAvite3/SHAvite_sha3.h
    hash_functions/SIMD/compat.h
    hash_functions/SIMD/Simd_sha3.h
    hash_functions/SIMD/tables.h
    hash_functions/Skein/brg_endian.h
    hash_functions/Skein/brg_types.h
    hash_functions/Skein/skein_debug.h
    hash_functions/Skein/skein.h
    hash_functions/Skein/skein_iv.h
    hash_functions/Skein/skein_port.h
    hash_functions/Skein/Skein_sha3.h
    hash_functions/SpectralHash/SpectralHash_sha3.h
    hash_functions/SpectralHash/spectral_structs.h
    hash_functions/StreamHash/sbox32.h
    hash_functions/StreamHash/StreamHash_sha3.h
    hash_functions/Tangle/Tangle_sha3.h
    hash_functions/Twister/Twister_sha3.h
    hash_functions/Twister/twister_tables.h
    hash_functions/Vortex/int_types.h
    hash_functions/Vortex/vortex_core.h
    hash_functions/Vortex/vortex_misc.h
    hash_functions/Vortex/Vortex_sha3.h
    hash_functions/WaMM/BitArray.h
    hash_functions/WaMM/ReverseBits.h
    hash_functions/WaMM/WaMMConstants.h
    hash_functions/WaMM/WaMMErrorMessage.h
    hash_functions/WaMM/WaMM.h
    hash_functions/WaMM/WaMMOperator.h
    hash_functions/WaMM/Wamm_sha3.h
    hash_functions/WaMM/WaMMTappingPrimes.h
    hash_functions/Waterfall/Waterfall_sha3.h
    #    hash_functions/ECOH/ecoh.h
    #    hash_functions/ECOH/Ecoh_sha3.h
    #    hash_functions/EnRUPT/EnRUPT_opt.h
    #    hash_functions/EnRUPT/Enrupt_sha3.h
    #    hash_functions/EnRUPT/portEnRUPT.h
    #    hash_functions/LUX/Lux_sha3.h
    #    hash_functions/SANDstorm/DoBlockModMix.h
    #    hash_functions/SANDstorm/Sandstorm_sha3.h
    #    hash_functions/SANDstorm/SHA3_ref.h
    #    hash_functions/SWIFFTX/inttypes.h
    #    hash_functions/SWIFFTX/stdbool.h
    #    hash_functions/SWIFFTX/stdint.h
    #    hash_functions/SWIFFTX/SWIFFTX.h
    #    hash_functions/SWIFFTX/Swifftx_sha3.h
    #    hash_functions/TIB3/inupfin.h
    #    hash_functions/TIB3/Tib_sha3.h
)


# === source groups ===
source_group(hash_functions     REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/hash_functions/.*")


add_eacirc_project(sha3 ${SOURCES} ${HEADERS})
target_link_libraries(sha3 tinyXML)