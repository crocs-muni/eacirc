TEMPLATE = app
CONFIG += console
CONFIG -= qt

SUPPRESSED_WARNINGS = -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable \
                      -Wno-unused-function -Wno-unused-value
unix{ QMAKE_CXX = g++-4.7 }

QMAKE_LFLAGS_RELEASE += -static -static-libgcc -static-libstdc++
QMAKE_CXXFLAGS += -std=c++11 -Wall -Wextra $$SUPPRESSED_WARNINGS # -Weffc++
QMAKE_CXXFLAGS += -isystem ../EACirc/galib -isystem ../EACirc/tinyXML
INCLUDEPATH += ./EACirc ./EACirc/galib ./EACirc/tinyXML

# === main project files ===
SOURCES += \
    EACirc/Main.cpp \
    EACirc/CircuitGenome.cpp \
    EACirc/CommonFnc.cpp \
    EACirc/XMLProcessor.cpp \
    EACirc/Logger.cpp \
    EACirc/Status.cpp \
    EACirc/EACirc.cpp \
    EACirc/EACglobals.cpp \

# === circuit processing ===
SOURCES += \
    EACirc/circuit/GACallbacks.cpp \

# === evaluators ===
SOURCES += \
    EACirc/evaluators/IEvaluator.cpp \
    EACirc/evaluators/TopBitEvaluator.cpp \
    EACirc/evaluators/CategoriesEvaluator.cpp \
    EACirc/evaluators/HammingWeightEvaluator.cpp \

# === random generators ===
SOURCES += \
    EACirc/generators/BiasRndGen.cpp \
    EACirc/generators/IRndGen.cpp \
    EACirc/generators/QuantumRndGen.cpp \
    EACirc/generators/MD5RndGen.cpp \
    EACirc/generators/md5.cpp \

# === testing using CATCH framework ===
SOURCES += \
    EACirc/self_tests/Tests.cpp \
    EACirc/self_tests/TestConfigurator.cpp \

# === project files ===
SOURCES += \
    EACirc/projects/IProject.cpp \
    EACirc/projects/pregenerated_tv/PregeneratedTvProject.cpp \
    EACirc/projects/sha3/Sha3Project.cpp \
    EACirc/projects/sha3/Hasher.cpp \
    EACirc/projects/sha3/Sha3Interface.cpp \
    EACirc/projects/estream/EstreamProject.cpp \
    EACirc/projects/estream/EncryptorDecryptor.cpp \
    EACirc/projects/estream/EstreamInterface.cpp \
    EACirc/projects/tea/TeaProject.cpp \
    EACirc/projects/files/filesProject.cpp \

# === eSTREAM cipher files ===
SOURCES += \
    EACirc/projects/estream/ciphers/zk-crypt/zk-crypt-v3.cpp \
    EACirc/projects/estream/ciphers/wg/wg.cpp \
    EACirc/projects/estream/ciphers/tsc-4/tsc-4.cpp \
    EACirc/projects/estream/ciphers/sosemanuk/sosemanuk.cpp \
    EACirc/projects/estream/ciphers/sfinks/sfinks.cpp \
    EACirc/projects/estream/ciphers/salsa20/salsa20.cpp \
    EACirc/projects/estream/ciphers/rabbit/rabbit.cpp \
    EACirc/projects/estream/ciphers/py/py6.cpp \
    EACirc/projects/estream/ciphers/pomaranch/pomaranch.cpp \
    EACirc/projects/estream/ciphers/mir-1/mir-1.cpp \
    EACirc/projects/estream/ciphers/mickey/mickey-128-v2.cpp \
    EACirc/projects/estream/ciphers/mag/mag.cpp \
    EACirc/projects/estream/ciphers/lex/lex.cpp \
    EACirc/projects/estream/ciphers/hermes/hermes.cpp \
    EACirc/projects/estream/ciphers/hc-128/hc-128.cpp \
    EACirc/projects/estream/ciphers/grain/grain-v1.cpp \
    EACirc/projects/estream/ciphers/fubuki/fubuki.cpp \
    EACirc/projects/estream/ciphers/ffcsr/f-fcsr-h.cpp \
    EACirc/projects/estream/ciphers/edon80/edon80.cpp \
    EACirc/projects/estream/ciphers/dragon/dragon.cpp \
    EACirc/projects/estream/ciphers/dragon/dragon-sboxes.cpp \
    EACirc/projects/estream/ciphers/dicing/dicing-v2.cpp \
    EACirc/projects/estream/ciphers/decim/decim-v2.cpp \
    EACirc/projects/estream/ciphers/cryptmt/cryptmt-v3.cpp \
    EACirc/projects/estream/ciphers/achterbahn/achterbahn-128-80.cpp \
    EACirc/projects/estream/ciphers/abc/abc-v3.cpp \
#    EACirc/projects/estream/ciphers/trivium/trivium.cpp \       # stopped working after IDE update
#    EACirc/projects/estream/ciphers/yamb/yamb.cpp \             # stopped working after IDE update
#    EACirc/projects/estream/ciphers/polarbear/polar-bear.cpp \  # not implemented in EncryptorDecryptor
#    EACirc/projects/estream/ciphers/polarbear/aescrypt.cpp \    # do not include!
#    EACirc/projects/estream/ciphers/polarbear/aestab.cpp \      # do not include!
#    EACirc/projects/estream/ciphers/polarbear/whirltab.cpp \    # do not include!
#    EACirc/projects/estream/ciphers/nls/nlsref.cpp \            # not implemented in EncryptorDecryptor
#    EACirc/projects/estream/ciphers/moustique/moustique.cpp \   # not implemented in EncryptorDecryptor
#    EACirc/projects/estream/ciphers/cryptmt/altivec.cpp \       # do not include!
#    EACirc/projects/estream/ciphers/cryptmt/sse2.cpp            # do not include!
#    EACirc/projects/estream/ciphers/decim/decim-128.c \         # do not include!

# === SHA-3 hash function files ===
SOURCES += \
    EACirc/projects/sha3/hash_functions/Abacus/Abacus_sha3.cpp \
    EACirc/projects/sha3/hash_functions/ARIRANG/Arirang_OP32.c \
    EACirc/projects/sha3/hash_functions/ARIRANG/Arirang_sha3.cpp \
    EACirc/projects/sha3/hash_functions/Aurora/Aurora_sha3.cpp \
    EACirc/projects/sha3/hash_functions/Blake/Blake_sha3.cpp \
    EACirc/projects/sha3/hash_functions/Blender/Blender_sha3.cpp \
    EACirc/projects/sha3/hash_functions/BMW/BMW_sha3.cpp \
    EACirc/projects/sha3/hash_functions/Boole/Boole_sha3.cpp \
    EACirc/projects/sha3/hash_functions/Cheetah/Cheetah_sha3.cpp \
    EACirc/projects/sha3/hash_functions/CHI/chi-fast32.c \
    EACirc/projects/sha3/hash_functions/CHI/Chi_sha3.cpp \
    EACirc/projects/sha3/hash_functions/CRUNCH/const_32.c \
    EACirc/projects/sha3/hash_functions/CRUNCH/crunch_224.c \
    EACirc/projects/sha3/hash_functions/CRUNCH/crunch_256.c \
    EACirc/projects/sha3/hash_functions/CRUNCH/crunch_384.c \
    EACirc/projects/sha3/hash_functions/CRUNCH/crunch_512.c \
    EACirc/projects/sha3/hash_functions/CRUNCH/Crunch_sha3.cpp \
    EACirc/projects/sha3/hash_functions/CubeHash/CubeHash_sha3.cpp \
    EACirc/projects/sha3/hash_functions/DCH/DCH_sha3.cpp \
    EACirc/projects/sha3/hash_functions/DynamicSHA2/DSHA2_sha3.cpp \
    EACirc/projects/sha3/hash_functions/DynamicSHA/DSHA_sha3.cpp \
    EACirc/projects/sha3/hash_functions/ECHO/Echo_sha3.cpp \
    EACirc/projects/sha3/hash_functions/EDON/Edon_sha3.cpp \
    EACirc/projects/sha3/hash_functions/ESSENCE/essence_compress_256.c \
    EACirc/projects/sha3/hash_functions/ESSENCE/essence_compress_512.c \
    EACirc/projects/sha3/hash_functions/ESSENCE/essence_L_tables.c \
    EACirc/projects/sha3/hash_functions/ESSENCE/Essence_sha3.cpp \
    EACirc/projects/sha3/hash_functions/Fugue/fugue_256.c \
    EACirc/projects/sha3/hash_functions/Fugue/fugue_384.c \
    EACirc/projects/sha3/hash_functions/Fugue/fugue_512.c \
    EACirc/projects/sha3/hash_functions/Fugue/fugue.c \
    EACirc/projects/sha3/hash_functions/Fugue/Fugue_sha3.cpp \
    EACirc/projects/sha3/hash_functions/Grostl/Grostl_sha3.cpp \
    EACirc/projects/sha3/hash_functions/Hamsi/hamsi-exp.c \
    EACirc/projects/sha3/hash_functions/Hamsi/Hamsi_sha3.cpp \
    EACirc/projects/sha3/hash_functions/Hamsi/hamsi-tables.c \
    EACirc/projects/sha3/hash_functions/Hamsi/i.hamsi-ref.c \
    EACirc/projects/sha3/hash_functions/JH/JH_sha3.cpp \
    EACirc/projects/sha3/hash_functions/Keccak/KeccakDuplex.c \
    EACirc/projects/sha3/hash_functions/Keccak/KeccakF-1600-opt32.c \
    EACirc/projects/sha3/hash_functions/Keccak/Keccak_sha3.cpp \
    EACirc/projects/sha3/hash_functions/Keccak/KeccakSponge.c \
    EACirc/projects/sha3/hash_functions/Khichidi/khichidi_core.cpp \
    EACirc/projects/sha3/hash_functions/Khichidi/Khichidi_sha3.cpp \
    EACirc/projects/sha3/hash_functions/Lane/Lane_sha3.cpp \
    EACirc/projects/sha3/hash_functions/Lesamnta/Lesamnta_sha3.cpp \
    EACirc/projects/sha3/hash_functions/Luffa/Luffa_sha3.cpp \
    EACirc/projects/sha3/hash_functions/MCSSHA3/Mcssha_sha3.cpp \
    EACirc/projects/sha3/hash_functions/MD6/md6_compress.c \
    EACirc/projects/sha3/hash_functions/MD6/md6_mode.c \
    EACirc/projects/sha3/hash_functions/MD6/MD6_sha3.cpp \
    EACirc/projects/sha3/hash_functions/MeshHash/MeshHash_sha3.cpp \
    EACirc/projects/sha3/hash_functions/NaSHA/Nasha_sha3.cpp \
    EACirc/projects/sha3/hash_functions/Sarmal/Sarmal_sha3.cpp \
    EACirc/projects/sha3/hash_functions/Shabal/Shabal_sha3.cpp \
    EACirc/projects/sha3/hash_functions/Shamata/Shamata_sha3.cpp \
    EACirc/projects/sha3/hash_functions/SHAvite3/SHAvite_sha3.cpp \
    EACirc/projects/sha3/hash_functions/SIMD/optimized.c \
    EACirc/projects/sha3/hash_functions/SIMD/Simd_sha3.cpp \
    EACirc/projects/sha3/hash_functions/Skein/skein_block.cpp \
    EACirc/projects/sha3/hash_functions/Skein/skein.cpp \
    EACirc/projects/sha3/hash_functions/Skein/skein_debug.cpp \
    EACirc/projects/sha3/hash_functions/Skein/Skein_sha3.cpp \
    EACirc/projects/sha3/hash_functions/SpectralHash/SpectralHash_sha3.cpp \
    EACirc/projects/sha3/hash_functions/StreamHash/StreamHash_sha3.cpp \
    EACirc/projects/sha3/hash_functions/Tangle/Tangle_sha3.cpp \
    EACirc/projects/sha3/hash_functions/Twister/Twister_sha3.cpp \
    EACirc/projects/sha3/hash_functions/Vortex/vortex_core.c \
    EACirc/projects/sha3/hash_functions/Vortex/vortex_misc.c \
    EACirc/projects/sha3/hash_functions/Vortex/Vortex_sha3.cpp \
    EACirc/projects/sha3/hash_functions/Vortex/vortex_tables.c \
    EACirc/projects/sha3/hash_functions/WaMM/BitArray.c \
    EACirc/projects/sha3/hash_functions/WaMM/ReverseBits.c \
    EACirc/projects/sha3/hash_functions/WaMM/WaMM.c \
    EACirc/projects/sha3/hash_functions/WaMM/WaMMErrorMessage.c \
    EACirc/projects/sha3/hash_functions/WaMM/WaMMOperator.c \
    EACirc/projects/sha3/hash_functions/WaMM/Wamm_sha3.cpp \
    EACirc/projects/sha3/hash_functions/WaMM/WaMMTappingPrimes.c \
    EACirc/projects/sha3/hash_functions/Waterfall/Waterfall_sha3.cpp \
#    EACirc/projects/sha3/hash_functions/ECOH/ecoh.param.cpp \
#    EACirc/projects/sha3/hash_functions/ECOH/Ecoh_sha3.cpp \
#    EACirc/projects/sha3/hash_functions/EnRUPT/EnRUPT_opt.c \
#    EACirc/projects/sha3/hash_functions/EnRUPT/Enrupt_sha3.cpp \
#    EACirc/projects/sha3/hash_functions/LUX/Lux_sha3.cpp \
#    EACirc/projects/sha3/hash_functions/SANDstorm/Sandstorm_sha3.cpp \
#    EACirc/projects/sha3/hash_functions/SWIFFTX/SWIFFTX.c \
#    EACirc/projects/sha3/hash_functions/SWIFFTX/Swifftx_sha3.cpp \
#    EACirc/projects/sha3/hash_functions/TIB3/inupfin256.c \
#    EACirc/projects/sha3/hash_functions/TIB3/inupfin512.c \
#    EACirc/projects/sha3/hash_functions/TIB3/Tib_sha3.cpp \

# === libraries ===
SOURCES += \
    EACirc/tinyXML/tinystr.cpp EACirc/tinyXML/tinyxml.cpp EACirc/tinyXML/tinyxmlerror.cpp EACirc/tinyXML/tinyxmlparser.cpp \
    EACirc/galib/GA1DArrayGenome.cpp EACirc/galib/GA1DBinStrGenome.cpp EACirc/galib/GA2DArrayGenome.cpp EACirc/galib/GA2DBinStrGenome.cpp \
    EACirc/galib/GA3DArrayGenome.cpp EACirc/galib/GA3DBinStrGenome.cpp EACirc/galib/GAAllele.cpp EACirc/galib/GABaseGA.cpp EACirc/galib/GABin2DecGenome.cpp \
    EACirc/galib/gabincvt.cpp EACirc/galib/GABinStr.cpp EACirc/galib/GADCrowdingGA.cpp EACirc/galib/GADemeGA.cpp EACirc/galib/gaerror.cpp EACirc/galib/GAGenome.cpp \
    EACirc/galib/GAIncGA.cpp EACirc/galib/GAList.cpp EACirc/galib/GAListBASE.cpp EACirc/galib/GAListGenome.cpp EACirc/galib/GAParameter.cpp EACirc/galib/GAPopulation.cpp \
    EACirc/galib/garandom.cpp EACirc/galib/GARealGenome.cpp EACirc/galib/GAScaling.cpp EACirc/galib/GASelector.cpp EACirc/galib/GASimpleGA.cpp EACirc/galib/GASStateGA.cpp \
    EACirc/galib/GAStatistics.cpp EACirc/galib/GAStringGenome.cpp EACirc/galib/GATree.cpp EACirc/galib/GATreeBASE.cpp EACirc/galib/GATreeGenome.cpp

# === main EACirc files ===
HEADERS += \
    EACirc/Main.h \
    EACirc/CircuitGenome.h \
    EACirc/CommonFnc.h \
    EACirc/XMLProcessor.h \
    EACirc/EACconstants.h \
    EACirc/EACglobals.h \
    EACirc/Logger.h \
    EACirc/Status.h \
    EACirc/EACirc.h \
    EACirc/Version.h \

# === circuit processing ===
HEADERS += \
    EACirc/circuit/GACallbacks.h \

# === standard evaluators ===
HEADERS += \
    EACirc/evaluators/IEvaluator.h \
    EACirc/evaluators/TopBitEvaluator.h \
    EACirc/evaluators/CategoriesEvaluator.h \
    EACirc/evaluators/HammingWeightEvaluator.h \

# === random generators ===
HEADERS += \
    EACirc/generators/BiasRndGen.h \
    EACirc/generators/IRndGen.h \
    EACirc/generators/QuantumRndGen.h \
    EACirc/generators/MD5RndGen.h \
    EACirc/generators/md5.h \

# === CATCH testing framework ===
HEADERS += \
    EACirc/self_tests/Catch.h \
    EACirc/self_tests/TestConfigurator.h \

# === project files ===
HEADERS += \
    EACirc/projects/IProject.h \
    EACirc/projects/pregenerated_tv/PregeneratedTvProject.h \
    EACirc/projects/sha3/Sha3Project.h \
    EACirc/projects/sha3/Sha3Interface.h \
    EACirc/projects/sha3/Sha3Constants.h \
    EACirc/projects/sha3/Hasher.h \
    EACirc/projects/estream/EstreamProject.h \
    EACirc/projects/estream/EncryptorDecryptor.h \
    EACirc/projects/estream/EstreamConstants.h \
    EACirc/projects/estream/EstreamInterface.h \
    EACirc/projects/tea/TeaProject.h \
    EACirc/projects/files/filesConstants.h \
    EACirc/projects/files/filesProject.h \

# === eSTREAM cipher files ===
HEADERS += \
    EACirc/projects/estream/ciphers/ecrypt-config.h \
    EACirc/projects/estream/ciphers/ecrypt-machine.h \
    EACirc/projects/estream/ciphers/ecrypt-portable.h \
    EACirc/projects/estream/ciphers/zk-crypt/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/zk-crypt/ZKdef.h \
    EACirc/projects/estream/ciphers/zk-crypt/ZKengine.h \
    EACirc/projects/estream/ciphers/yamb/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/wg/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/tsc-4/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/trivium/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/sosemanuk/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/sosemanuk/sosemanuk.h \
    EACirc/projects/estream/ciphers/sfinks/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/salsa20/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/rabbit/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/py/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/pomaranch/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/pomaranch/pomaranch.h \
    EACirc/projects/estream/ciphers/mir-1/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/mir-1/mir1.h \
    EACirc/projects/estream/ciphers/mickey/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/mag/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/mag/unrolliv.h \
    EACirc/projects/estream/ciphers/mag/unrollmain.h \
    EACirc/projects/estream/ciphers/lex/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/hermes/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/hc-128/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/grain/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/grain/grain-v1.h \
    EACirc/projects/estream/ciphers/fubuki/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/ffcsr/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/ffcsr/ffcsrh-sync.h \
    EACirc/projects/estream/ciphers/edon80/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/dragon/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/dicing/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/decim/decimv2.h \
    EACirc/projects/estream/ciphers/decim/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/cryptmt/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/cryptmt/params.h \
    EACirc/projects/estream/ciphers/achterbahn/achterbahn.h \
    EACirc/projects/estream/ciphers/achterbahn/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/abc/abc.h \
    EACirc/projects/estream/ciphers/abc/abc-tables.h \
    EACirc/projects/estream/ciphers/abc/ecrypt-sync.h \
#    EACirc/projects/estream/ciphers/polarbear/aes.h \           # not implemented in EncryptorDecryptor
#    EACirc/projects/estream/ciphers/polarbear/aesopt.h \        # not implemented in EncryptorDecryptor
#    EACirc/projects/estream/ciphers/polarbear/ecrypt-sync.h \   # not implemented in EncryptorDecryptor
#    EACirc/projects/estream/ciphers/nls/ecrypt-sync.h \         # not implemented in EncryptorDecryptor
#    EACirc/projects/estream/ciphers/nls/ecrypt-sync-ae.h \      # not implemented in EncryptorDecryptor
#    EACirc/projects/estream/ciphers/nls/nls.h \                 # not implemented in EncryptorDecryptor
#    EACirc/projects/estream/ciphers/nls/nlsmultab.h \           # not implemented in EncryptorDecryptor
#    EACirc/projects/estream/ciphers/nls/nlssbox.h \             # not implemented in EncryptorDecryptor
#    EACirc/projects/estream/ciphers/moustique/ecrypt-ssyn.h \   # not implemented in EncryptorDecryptor
#    EACirc/projects/estream/ciphers/decim/decim-128.h           # do not include!
# not used (but included):
#    EACirc/projects/estream/ciphers/zk-crypt/ecrypt-sync.h \    # not used
#    EACirc/projects/estream/ciphers/zk-crypt/ZKdef.h \          # not used
#    EACirc/projects/estream/ciphers/zk-crypt/ZKengine.h \       # not used
#    EACirc/projects/estream/ciphers/cryptmt/ecrypt-sync.h \     # not used
#    EACirc/projects/estream/ciphers/cryptmt/params.h \          # not used
#    EACirc/projects/estream/ciphers/abc/abc.h \                 # not used
#    EACirc/projects/estream/ciphers/abc/abc-tables.h \          # not used
#    EACirc/projects/estream/ciphers/abc/ecrypt-sync.h \         # not used

# === SHA-3 hash function files ===
HEADERS += \
    EACirc/projects/sha3/hash_functions/hashFunctions.h \
    EACirc/projects/sha3/hash_functions/Abacus/Abacus_sha3.h \
    EACirc/projects/sha3/hash_functions/ARIRANG/Arirang_OP32.h \
    EACirc/projects/sha3/hash_functions/ARIRANG/Arirang_sha3.h \
    EACirc/projects/sha3/hash_functions/Aurora/Aurora_sha3.h \
    EACirc/projects/sha3/hash_functions/Blake/Blake_sha3.h \
    EACirc/projects/sha3/hash_functions/Blender/Blender_sha3.h \
    EACirc/projects/sha3/hash_functions/BMW/BMW_sha3.h \
    EACirc/projects/sha3/hash_functions/Boole/Boole_sha3.h \
    EACirc/projects/sha3/hash_functions/Cheetah/Cheetah_sha3.h \
    EACirc/projects/sha3/hash_functions/CHI/chi.h \
    EACirc/projects/sha3/hash_functions/CHI/Chi_sha3.h \
    EACirc/projects/sha3/hash_functions/CRUNCH/crunch_224.h \
    EACirc/projects/sha3/hash_functions/CRUNCH/crunch_256.h \
    EACirc/projects/sha3/hash_functions/CRUNCH/crunch_384.h \
    EACirc/projects/sha3/hash_functions/CRUNCH/crunch_512.h \
    EACirc/projects/sha3/hash_functions/CRUNCH/Crunch_sha3.h \
    EACirc/projects/sha3/hash_functions/CRUNCH/crunch_type.h \
    EACirc/projects/sha3/hash_functions/CubeHash/CubeHash_sha3.h \
    EACirc/projects/sha3/hash_functions/DCH/DCH_sha3.h \
    EACirc/projects/sha3/hash_functions/DynamicSHA2/DSHA2_sha3.h \
    EACirc/projects/sha3/hash_functions/DynamicSHA/DSHA_sha3.h \
    EACirc/projects/sha3/hash_functions/ECHO/Echo_sha3.h \
#    EACirc/projects/sha3/hash_functions/ECOH/ecoh.h \
#    EACirc/projects/sha3/hash_functions/ECOH/Ecoh_sha3.h \
    EACirc/projects/sha3/hash_functions/EDON/Edon_sha3.h \
#    EACirc/projects/sha3/hash_functions/EnRUPT/EnRUPT_opt.h \
#    EACirc/projects/sha3/hash_functions/EnRUPT/Enrupt_sha3.h \
#    EACirc/projects/sha3/hash_functions/EnRUPT/portEnRUPT.h \
    EACirc/projects/sha3/hash_functions/ESSENCE/essence.h \
    EACirc/projects/sha3/hash_functions/ESSENCE/Essence_sha3.h \
    EACirc/projects/sha3/hash_functions/Fugue/aestab.h \
    EACirc/projects/sha3/hash_functions/Fugue/aestab_t.h \
    EACirc/projects/sha3/hash_functions/Fugue/fugue_256.h \
    EACirc/projects/sha3/hash_functions/Fugue/fugue_384.h \
    EACirc/projects/sha3/hash_functions/Fugue/fugue_512.h \
    EACirc/projects/sha3/hash_functions/Fugue/fugue.h \
    EACirc/projects/sha3/hash_functions/Fugue/Fugue_sha3.h \
    EACirc/projects/sha3/hash_functions/Fugue/fugue_t.h \
    EACirc/projects/sha3/hash_functions/Grostl/brg_endian.h \
    EACirc/projects/sha3/hash_functions/Grostl/brg_types.h \
    EACirc/projects/sha3/hash_functions/Grostl/Grostl_sha3.h \
    EACirc/projects/sha3/hash_functions/Grostl/tables.h \
    EACirc/projects/sha3/hash_functions/Hamsi/hamsi.h \
    EACirc/projects/sha3/hash_functions/Hamsi/hamsi-internals.h \
    EACirc/projects/sha3/hash_functions/Hamsi/Hamsi_sha3.h \
    EACirc/projects/sha3/hash_functions/Hamsi/hamsi-tables.h \
    EACirc/projects/sha3/hash_functions/JH/JH_sha3.h \
    EACirc/projects/sha3/hash_functions/Keccak/KeccakSponge.h \
    EACirc/projects/sha3/hash_functions/Keccak/brg_endian.h \
    EACirc/projects/sha3/hash_functions/Keccak/KeccakDuplex.h \
    EACirc/projects/sha3/hash_functions/Keccak/KeccakF-1600-interface.h \
    EACirc/projects/sha3/hash_functions/Keccak/KeccakF-1600-int-set.h \
    EACirc/projects/sha3/hash_functions/Keccak/KeccakF-1600-opt32-settings.h \
    EACirc/projects/sha3/hash_functions/Keccak/Keccak_sha3.h \
    EACirc/projects/sha3/hash_functions/Khichidi/common.h \
    EACirc/projects/sha3/hash_functions/Khichidi/khichidi_core.h \
    EACirc/projects/sha3/hash_functions/Khichidi/Khichidi_sha3.h \
    EACirc/projects/sha3/hash_functions/Lane/Lane_sha3.h \
    EACirc/projects/sha3/hash_functions/Lesamnta/Lesamnta_sha3.h \
    EACirc/projects/sha3/hash_functions/Luffa/Luffa_sha3.h \
#    EACirc/projects/sha3/hash_functions/LUX/Lux_sha3.h \
    EACirc/projects/sha3/hash_functions/MCSSHA3/Mcssha_sha3.h \
    EACirc/projects/sha3/hash_functions/MD6/inttypes.h \
    EACirc/projects/sha3/hash_functions/MD6/md6.h \
    EACirc/projects/sha3/hash_functions/MD6/MD6_sha3.h \
    EACirc/projects/sha3/hash_functions/MD6/stdint.h \
    EACirc/projects/sha3/hash_functions/MeshHash/MeshHash_sha3.h \
    EACirc/projects/sha3/hash_functions/NaSHA/brg_endian.h \
    EACirc/projects/sha3/hash_functions/NaSHA/brg_types.h \
    EACirc/projects/sha3/hash_functions/NaSHA/Nasha_sha3.h \
#    EACirc/projects/sha3/hash_functions/SANDstorm/DoBlockModMix.h \
#    EACirc/projects/sha3/hash_functions/SANDstorm/Sandstorm_sha3.h \
#    EACirc/projects/sha3/hash_functions/SANDstorm/SHA3_ref.h \
    EACirc/projects/sha3/hash_functions/Sarmal/Sarmal_sha3.h \
    EACirc/projects/sha3/hash_functions/Shabal/Shabal_sha3.h \
    EACirc/projects/sha3/hash_functions/Shamata/Shamata_sha3.h \
    EACirc/projects/sha3/hash_functions/SHAvite3/AESround.h \
    EACirc/projects/sha3/hash_functions/SHAvite3/portable.h \
    EACirc/projects/sha3/hash_functions/SHAvite3/SHAvite3-256.h \
    EACirc/projects/sha3/hash_functions/SHAvite3/SHAvite3-512.h \
    EACirc/projects/sha3/hash_functions/SHAvite3/SHAvite_sha3.h \
    EACirc/projects/sha3/hash_functions/SIMD/compat.h \
    EACirc/projects/sha3/hash_functions/SIMD/Simd_sha3.h \
    EACirc/projects/sha3/hash_functions/SIMD/tables.h \
    EACirc/projects/sha3/hash_functions/Skein/brg_endian.h \
    EACirc/projects/sha3/hash_functions/Skein/brg_types.h \
    EACirc/projects/sha3/hash_functions/Skein/skein_debug.h \
    EACirc/projects/sha3/hash_functions/Skein/skein.h \
    EACirc/projects/sha3/hash_functions/Skein/skein_iv.h \
    EACirc/projects/sha3/hash_functions/Skein/skein_port.h \
    EACirc/projects/sha3/hash_functions/Skein/Skein_sha3.h \
    EACirc/projects/sha3/hash_functions/SpectralHash/SpectralHash_sha3.h \
    EACirc/projects/sha3/hash_functions/SpectralHash/spectral_structs.h \
    EACirc/projects/sha3/hash_functions/StreamHash/sbox32.h \
    EACirc/projects/sha3/hash_functions/StreamHash/StreamHash_sha3.h \
#    EACirc/projects/sha3/hash_functions/SWIFFTX/inttypes.h \
#    EACirc/projects/sha3/hash_functions/SWIFFTX/stdbool.h \
#    EACirc/projects/sha3/hash_functions/SWIFFTX/stdint.h \
#    EACirc/projects/sha3/hash_functions/SWIFFTX/SWIFFTX.h \
#    EACirc/projects/sha3/hash_functions/SWIFFTX/Swifftx_sha3.h \
    EACirc/projects/sha3/hash_functions/Tangle/Tangle_sha3.h \
#    EACirc/projects/sha3/hash_functions/TIB3/inupfin.h \
#    EACirc/projects/sha3/hash_functions/TIB3/Tib_sha3.h \
    EACirc/projects/sha3/hash_functions/Twister/Twister_sha3.h \
    EACirc/projects/sha3/hash_functions/Twister/twister_tables.h \
    EACirc/projects/sha3/hash_functions/Vortex/int_types.h \
    EACirc/projects/sha3/hash_functions/Vortex/vortex_core.h \
    EACirc/projects/sha3/hash_functions/Vortex/vortex_misc.h \
    EACirc/projects/sha3/hash_functions/Vortex/Vortex_sha3.h \
    EACirc/projects/sha3/hash_functions/WaMM/BitArray.h \
    EACirc/projects/sha3/hash_functions/WaMM/ReverseBits.h \
    EACirc/projects/sha3/hash_functions/WaMM/WaMMConstants.h \
    EACirc/projects/sha3/hash_functions/WaMM/WaMMErrorMessage.h \
    EACirc/projects/sha3/hash_functions/WaMM/WaMM.h \
    EACirc/projects/sha3/hash_functions/WaMM/WaMMOperator.h \
    EACirc/projects/sha3/hash_functions/WaMM/Wamm_sha3.h \
    EACirc/projects/sha3/hash_functions/WaMM/WaMMTappingPrimes.h \
    EACirc/projects/sha3/hash_functions/Waterfall/Waterfall_sha3.h \

# === Libraries (redundant if using pre-compiled) ===
HEADERS += \
    EACirc/tinyXML/tinystr.h EACirc/tinyXML/tinyxml.h \
    EACirc/galib/GA1DArrayGenome.h EACirc/galib/ga.h EACirc/galib/GA1DBinStrGenome.h EACirc/galib/GA2DArrayGenome.h EACirc/galib/GA2DBinStrGenome.h \
    EACirc/galib/GA3DArrayGenome.h EACirc/galib/GA3DBinStrGenome.h EACirc/galib/GAAllele.h EACirc/galib/GAArray.h EACirc/galib/GABaseGA.h \
    EACirc/galib/GABin2DecGenome.h EACirc/galib/gabincvt.h EACirc/galib/GABinStr.h EACirc/galib/gaconfig.h EACirc/galib/GADCrowdingGA.h EACirc/galib/GADemeGA.h \
    EACirc/galib/gaerror.h EACirc/galib/GAEvalData.h EACirc/galib/GAGenome.h EACirc/galib/gaid.h EACirc/galib/GAIncGA.h EACirc/galib/GAList.h EACirc/galib/GAListBASE.h \
    EACirc/galib/GAListGenome.h EACirc/galib/GAMask.h EACirc/galib/GANode.h EACirc/galib/GAParameter.h EACirc/galib/GAPopulation.h EACirc/galib/garandom.h \
    EACirc/galib/GARealGenome.h EACirc/galib/GAScaling.h EACirc/galib/GASelector.h EACirc/galib/GASimpleGA.h EACirc/galib/GASStateGA.h EACirc/galib/GAStatistics.h \
    EACirc/galib/GAStringGenome.h EACirc/galib/GATree.h EACirc/galib/GATreeBASE.h EACirc/galib/GATreeGenome.h EACirc/galib/gatypes.h EACirc/galib/gaversion.h \
    EACirc/galib/std_stream.h

OTHER_FILES += \
    EACirc/config.xml
