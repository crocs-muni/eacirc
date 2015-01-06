TEMPLATE=app
CONFIG+=console
CONFIG-=qt
TARGET=EACirc

SUPPRESSED_WARNINGS = -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable \
    -Wno-unused-function -Wno-unused-value

QMAKE_TARGET = EACirc
QMAKE_LFLAGS_RELEASE += # -static -static-libgcc -static-libstdc++
QMAKE_CXXFLAGS += -std=c++11 -Wall -Wextra $$SUPPRESSED_WARNINGS # -Weffc++
QMAKE_CXXFLAGS += -isystem ../EACirc/galib -isystem ../EACirc/tinyXML
INCLUDEPATH += ./EACirc ./EACirc/galib ./EACirc/tinyXML
LIBS += -lcrypto

# === main project files ===
SOURCES += \
    EACirc/Main.cpp \
    EACirc/CommonFnc.cpp \
    EACirc/XMLProcessor.cpp \
    EACirc/Logger.cpp \
    EACirc/Status.cpp \
    EACirc/EACirc.cpp \
    EACirc/EACglobals.cpp \

# === individual representation ===
SOURCES += \
    EACirc/circuit/ICircuit.cpp \
    EACirc/circuit/ICircuitIO.cpp \

# === circuit processing ===
SOURCES += \
    EACirc/circuit/gate/GateCircuitIO.cpp \
    EACirc/circuit/gate/GateCircuit.cpp \
    EACirc/circuit/gate/GACallbacks.cpp \
    EACirc/circuit/gate/CircuitInterpreter.cpp \
    EACirc/circuit/gate/CircuitCommonFunctions.cpp \

# === polynomial circuits ===
SOURCES += \
    EACirc/circuit/polynomial/PolynomialCircuit.cpp \
    EACirc/circuit/polynomial/PolynomialCircuitIO.cpp \
    EACirc/circuit/polynomial/Term.cpp   \
    EACirc/circuit/polynomial/PolyDistEval.cpp   \
    EACirc/circuit/polynomial/GAPolyCallbacks.cpp \
    EACirc/circuit/polynomial/poly.cpp \

# === evaluators ===
SOURCES += \
    EACirc/evaluators/IEvaluator.cpp \
    EACirc/evaluators/TopBitEvaluator.cpp \
    EACirc/evaluators/CategoriesEvaluator.cpp \
    EACirc/evaluators/HammingWeightEvaluator.cpp \
    EACirc/evaluators/FeatureEvaluator.cpp \

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
    EACirc/projects/files/filesProject.cpp \
    EACirc/projects/caesar/CaesarProject.cpp \
    EACirc/projects/caesar/CaesarInterface.cpp \
    EACirc/projects/caesar/Encryptor.cpp \

# === CAESAR algorithms ===
SOURCES += \
    EACirc/projects/caesar/aead/common/crypto_verify_16.cpp \
    EACirc/projects/caesar/aead/common/crypto_core_aes128encrypt.cpp \

# === eSTREAM cipher files ===
SOURCES += \
    EACirc/projects/estream/ciphers/zk-crypt/zk-crypt-v3.cpp \
    EACirc/projects/estream/ciphers/wg/wg.cpp \
    EACirc/projects/estream/ciphers/tsc-4/tsc-4.cpp \
    EACirc/projects/estream/ciphers/tea/tea.cpp \
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
    EACirc/CommonFnc.h \
    EACirc/XMLProcessor.h \
    EACirc/EACconstants.h \
    EACirc/EACglobals.h \
    EACirc/Logger.h \
    EACirc/Status.h \
    EACirc/EACirc.h \
    EACirc/Version.h \

# === individual representation ===
HEADERS += \
    EACirc/circuit/ICircuit.h \
    EACirc/circuit/ICircuitIO.h \

# === circuit processing ===
HEADERS += \
    EACirc/circuit/gate/GACallbacks.h \
    EACirc/circuit/gate/CircuitInterpreter.h \
    EACirc/circuit/gate/CircuitCommonFunctions.h \
    EACirc/circuit/gate/GateCircuit.h \
    EACirc/circuit/gate/GateCircuitIO.h \

# === polynomials ===
HEADERS += \
    EACirc/circuit/polynomial/PolyDistEval.h \
    EACirc/circuit/polynomial/GAPolyCallbacks.h \
    EACirc/circuit/polynomial/Term.h \
    EACirc/circuit/polynomial/poly.h \
    EACirc/circuit/polynomial/PolynomialCircuit.h \
    EACirc/circuit/polynomial/PolynomialCircuitIO.h \

# === standard evaluators ===
HEADERS += \
    EACirc/evaluators/IEvaluator.h \
    EACirc/evaluators/TopBitEvaluator.h \
    EACirc/evaluators/CategoriesEvaluator.h \
    EACirc/evaluators/HammingWeightEvaluator.h \
    EACirc/evaluators/FeatureEvaluator.h \

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
    EACirc/projects/files/filesConstants.h \
    EACirc/projects/files/filesProject.h \
    EACirc/projects/caesar/CaesarConstants.h \
    EACirc/projects/caesar/CaesarProject.h \
    EACirc/projects/caesar/CaesarInterface.h \
    EACirc/projects/caesar/Encryptor.h \

# === CAESAR algorithms ===
HEADERS += \
    EACirc/projects/caesar/aead/common/api.h \
    EACirc/projects/caesar/aead/aead.h \

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
    EACirc/projects/estream/ciphers/tea/ecrypt-sync.h \
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
    EACirc/projects/estream/ciphers/drag-on/ecrypt-sync.h \
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
    EACirc/projects/sha3/hash_functions/EDON/Edon_sha3.h \
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
    EACirc/projects/sha3/hash_functions/MCSSHA3/Mcssha_sha3.h \
    EACirc/projects/sha3/hash_functions/MD6/inttypes.h \
    EACirc/projects/sha3/hash_functions/MD6/md6.h \
    EACirc/projects/sha3/hash_functions/MD6/MD6_sha3.h \
    EACirc/projects/sha3/hash_functions/MD6/stdint.h \
    EACirc/projects/sha3/hash_functions/MeshHash/MeshHash_sha3.h \
    EACirc/projects/sha3/hash_functions/NaSHA/brg_endian.h \
    EACirc/projects/sha3/hash_functions/NaSHA/brg_types.h \
    EACirc/projects/sha3/hash_functions/NaSHA/Nasha_sha3.h \
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
    EACirc/projects/sha3/hash_functions/Tangle/Tangle_sha3.h \
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
#    EACirc/projects/sha3/hash_functions/ECOH/ecoh.h \
#    EACirc/projects/sha3/hash_functions/ECOH/Ecoh_sha3.h \
#    EACirc/projects/sha3/hash_functions/EnRUPT/EnRUPT_opt.h \
#    EACirc/projects/sha3/hash_functions/EnRUPT/Enrupt_sha3.h \
#    EACirc/projects/sha3/hash_functions/EnRUPT/portEnRUPT.h \
#    EACirc/projects/sha3/hash_functions/LUX/Lux_sha3.h \
#    EACirc/projects/sha3/hash_functions/SANDstorm/DoBlockModMix.h \
#    EACirc/projects/sha3/hash_functions/SANDstorm/Sandstorm_sha3.h \
#    EACirc/projects/sha3/hash_functions/SANDstorm/SHA3_ref.h \
#    EACirc/projects/sha3/hash_functions/SWIFFTX/inttypes.h \
#    EACirc/projects/sha3/hash_functions/SWIFFTX/stdbool.h \
#    EACirc/projects/sha3/hash_functions/SWIFFTX/stdint.h \
#    EACirc/projects/sha3/hash_functions/SWIFFTX/SWIFFTX.h \
#    EACirc/projects/sha3/hash_functions/SWIFFTX/Swifftx_sha3.h \
#    EACirc/projects/sha3/hash_functions/TIB3/inupfin.h \
#    EACirc/projects/sha3/hash_functions/TIB3/Tib_sha3.h \

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
    EACirc/config.xml \

# CAESAR candidates info files
OTHER_FILES += \
    EACirc/projects/caesar/aead/common/About.md \

SOURCES += \
 EACirc/projects/caesar/aead/acorn128/acorn128_encrypt.cpp \
 EACirc/projects/caesar/aead/acorn128/Acorn128.cpp \
 EACirc/projects/caesar/aead/aeadaes128ocbtaglen128v1/aeadaes128ocbtaglen128v1_encrypt.cpp \
 EACirc/projects/caesar/aead/aeadaes128ocbtaglen128v1/Aeadaes128ocbtaglen128v1.cpp \
 EACirc/projects/caesar/aead/aeadaes128ocbtaglen64v1/aeadaes128ocbtaglen64v1_encrypt.cpp \
 EACirc/projects/caesar/aead/aeadaes128ocbtaglen64v1/Aeadaes128ocbtaglen64v1.cpp \
 EACirc/projects/caesar/aead/aeadaes128ocbtaglen96v1/aeadaes128ocbtaglen96v1_encrypt.cpp \
 EACirc/projects/caesar/aead/aeadaes128ocbtaglen96v1/Aeadaes128ocbtaglen96v1.cpp \
 EACirc/projects/caesar/aead/aeadaes192ocbtaglen128v1/aeadaes192ocbtaglen128v1_encrypt.cpp \
 EACirc/projects/caesar/aead/aeadaes192ocbtaglen128v1/Aeadaes192ocbtaglen128v1.cpp \
 EACirc/projects/caesar/aead/aeadaes192ocbtaglen64v1/aeadaes192ocbtaglen64v1_encrypt.cpp \
 EACirc/projects/caesar/aead/aeadaes192ocbtaglen64v1/Aeadaes192ocbtaglen64v1.cpp \
 EACirc/projects/caesar/aead/aeadaes192ocbtaglen96v1/aeadaes192ocbtaglen96v1_encrypt.cpp \
 EACirc/projects/caesar/aead/aeadaes192ocbtaglen96v1/Aeadaes192ocbtaglen96v1.cpp \
 EACirc/projects/caesar/aead/aeadaes256ocbtaglen128v1/aeadaes256ocbtaglen128v1_encrypt.cpp \
 EACirc/projects/caesar/aead/aeadaes256ocbtaglen128v1/Aeadaes256ocbtaglen128v1.cpp \
 EACirc/projects/caesar/aead/aeadaes256ocbtaglen64v1/aeadaes256ocbtaglen64v1_encrypt.cpp \
 EACirc/projects/caesar/aead/aeadaes256ocbtaglen64v1/Aeadaes256ocbtaglen64v1.cpp \
 EACirc/projects/caesar/aead/aeadaes256ocbtaglen96v1/aeadaes256ocbtaglen96v1_encrypt.cpp \
 EACirc/projects/caesar/aead/aeadaes256ocbtaglen96v1/Aeadaes256ocbtaglen96v1.cpp \
 EACirc/projects/caesar/aead/aegis128/aegis128_aes.cpp \
 EACirc/projects/caesar/aead/aegis128/aegis128_encrypt.cpp \
 EACirc/projects/caesar/aead/aegis128/Aegis128.cpp \
 EACirc/projects/caesar/aead/aegis128l/aegis128l_aes.cpp \
 EACirc/projects/caesar/aead/aegis128l/aegis128l_ecrypt.cpp \
 EACirc/projects/caesar/aead/aegis128l/Aegis128l.cpp \
 EACirc/projects/caesar/aead/aegis256/aegis256_aes.cpp \
 EACirc/projects/caesar/aead/aegis256/aegis256_encrypt.cpp \
 EACirc/projects/caesar/aead/aegis256/Aegis256.cpp \
# EACirc/projects/caesar/aead/aes128avalanchev1/aes128avalanchev1_avalanche.cpp \ # compilaation ok, sigsegv while running
# EACirc/projects/caesar/aead/aes128avalanchev1/aes128avalanchev1_largenumbers.cpp \ # compilaation ok, sigsegv while running
# EACirc/projects/caesar/aead/aes128avalanchev1/aes128avalanchev1_pcmac.cpp \ # compilaation ok, sigsegv while running
# EACirc/projects/caesar/aead/aes128avalanchev1/aes128avalanchev1_rmac.cpp \ # compilaation ok, sigsegv while running
# EACirc/projects/caesar/aead/aes128avalanchev1/Aes128avalanchev1.cpp \ # compilaation ok, sigsegv while running
# EACirc/projects/caesar/aead/aes128cpfbv1/aes128cpfbv1_encrypt.cpp \
# EACirc/projects/caesar/aead/aes128cpfbv1/Aes128cpfbv1.cpp \

SOURCES += \
 EACirc/projects/caesar/aead/aes128gcmv1/aes128gcmv1_encrypt.cpp \
 EACirc/projects/caesar/aead/aes128gcmv1/Aes128gcmv1.cpp \
 EACirc/projects/caesar/aead/aes128marble4rv1/aes128marble4rv1_aes_core.cpp \
 EACirc/projects/caesar/aead/aes128marble4rv1/aes128marble4rv1_encrypt.cpp \
 EACirc/projects/caesar/aead/aes128marble4rv1/aes128marble4rv1_marble.cpp \
 EACirc/projects/caesar/aead/aes128marble4rv1/aes128marble4rv1_utils.cpp \
 EACirc/projects/caesar/aead/aes128marble4rv1/Aes128marble4rv1.cpp \
 EACirc/projects/caesar/aead/aes128n12clocv1/aes128n12clocv1_aes_core.cpp \
 EACirc/projects/caesar/aead/aes128n12clocv1/aes128n12clocv1_cloc.cpp \
 EACirc/projects/caesar/aead/aes128n12clocv1/aes128n12clocv1_encrypt.cpp \
 EACirc/projects/caesar/aead/aes128n12clocv1/aes128n12clocv1_utils.cpp \
 EACirc/projects/caesar/aead/aes128n12clocv1/Aes128n12clocv1.cpp \
# EACirc/projects/caesar/aead/aes128n12silcv1/aes128n12silcv1_aes_core.cpp \
# EACirc/projects/caesar/aead/aes128n12silcv1/aes128n12silcv1_encrypt.cpp \
# EACirc/projects/caesar/aead/aes128n12silcv1/aes128n12silcv1_silc.cpp \
# EACirc/projects/caesar/aead/aes128n12silcv1/aes128n12silcv1_utils.cpp \
# EACirc/projects/caesar/aead/aes128n12silcv1/Aes128n12silcv1.cpp \

SOURCES += \
 EACirc/projects/caesar/aead/aes128n8clocv1/aes128n8clocv1_aes_core.cpp \
 EACirc/projects/caesar/aead/aes128n8clocv1/aes128n8clocv1_cloc.cpp \
 EACirc/projects/caesar/aead/aes128n8clocv1/aes128n8clocv1_encrypt.cpp \
 EACirc/projects/caesar/aead/aes128n8clocv1/aes128n8clocv1_utils.cpp \
 EACirc/projects/caesar/aead/aes128n8clocv1/Aes128n8clocv1.cpp \
# EACirc/projects/caesar/aead/aes128n8silcv1/aes128n8silcv1_aes_core.cpp \
# EACirc/projects/caesar/aead/aes128n8silcv1/aes128n8silcv1_encrypt.cpp \
# EACirc/projects/caesar/aead/aes128n8silcv1/aes128n8silcv1_silc.cpp \
# EACirc/projects/caesar/aead/aes128n8silcv1/aes128n8silcv1_utils.cpp \
# EACirc/projects/caesar/aead/aes128n8silcv1/Aes128n8silcv1.cpp \

SOURCES += \
 EACirc/projects/caesar/aead/aes128otrpv1/aes128otrpv1_encrypt.cpp \
 EACirc/projects/caesar/aead/aes128otrpv1/aes128otrpv1_OTR.cpp \
 EACirc/projects/caesar/aead/aes128otrpv1/aes128otrpv1_t-aes_enc_only.cpp \
 EACirc/projects/caesar/aead/aes128otrpv1/Aes128otrpv1.cpp \
 EACirc/projects/caesar/aead/aes128otrsv1/aes128otrsv1_encrypt.cpp \
 EACirc/projects/caesar/aead/aes128otrsv1/aes128otrsv1_OTR.cpp \
 EACirc/projects/caesar/aead/aes128otrsv1/aes128otrsv1_t-aes_enc_only.cpp \
 EACirc/projects/caesar/aead/aes128otrsv1/Aes128otrsv1.cpp \
 EACirc/projects/caesar/aead/aes128poetv1aes128/aes128poetv1aes128_aes.cpp \
 EACirc/projects/caesar/aead/aes128poetv1aes128/aes128poetv1aes128_encrypt.cpp \
 EACirc/projects/caesar/aead/aes128poetv1aes128/aes128poetv1aes128_poet.cpp \
 EACirc/projects/caesar/aead/aes128poetv1aes128/Aes128poetv1aes128.cpp \
 EACirc/projects/caesar/aead/aes128poetv1aes4/aes128poetv1aes4_aes.cpp \
 EACirc/projects/caesar/aead/aes128poetv1aes4/aes128poetv1aes4_encrypt.cpp \
 EACirc/projects/caesar/aead/aes128poetv1aes4/aes128poetv1aes4_poet.cpp \
 EACirc/projects/caesar/aead/aes128poetv1aes4/Aes128poetv1aes4.cpp \
# EACirc/projects/caesar/aead/aes192avalanchev1/aes192avalanchev1_avalanche.cpp \
# EACirc/projects/caesar/aead/aes192avalanchev1/aes192avalanchev1_largenumbers.cpp \
# EACirc/projects/caesar/aead/aes192avalanchev1/aes192avalanchev1_pcmac.cpp \
# EACirc/projects/caesar/aead/aes192avalanchev1/aes192avalanchev1_rmac.cpp \
# EACirc/projects/caesar/aead/aes192avalanchev1/Aes192avalanchev1.cpp \
# EACirc/projects/caesar/aead/aes256avalanchev1/aes256avalanchev1_avalanche.cpp \
# EACirc/projects/caesar/aead/aes256avalanchev1/aes256avalanchev1_largenumbers.cpp \
# EACirc/projects/caesar/aead/aes256avalanchev1/aes256avalanchev1_pcmac.cpp \
# EACirc/projects/caesar/aead/aes256avalanchev1/aes256avalanchev1_rmac.cpp \
# EACirc/projects/caesar/aead/aes256avalanchev1/Aes256avalanchev1.cpp \

SOURCES += \
# EACirc/projects/caesar/aead/aes256cpfbv1/aes256cpfbv1_encrypt.cpp \
# EACirc/projects/caesar/aead/aes256cpfbv1/Aes256cpfbv1.cpp \
# EACirc/projects/caesar/aead/aes256gcmv1/aes256gcmv1_encrypt.cpp \
# EACirc/projects/caesar/aead/aes256gcmv1/Aes256gcmv1.cpp \
# EACirc/projects/caesar/aead/aes256otrpv1/aes256otrpv1_encrypt.cpp \
# EACirc/projects/caesar/aead/aes256otrpv1/aes256otrpv1_OTR.cpp \
# EACirc/projects/caesar/aead/aes256otrpv1/aes256otrpv1_t-aes_enc_only.cpp \
# EACirc/projects/caesar/aead/aes256otrpv1/Aes256otrpv1.cpp \
# EACirc/projects/caesar/aead/aes256otrsv1/aes256otrsv1_encrypt.cpp \
# EACirc/projects/caesar/aead/aes256otrsv1/aes256otrsv1_OTR.cpp \
# EACirc/projects/caesar/aead/aes256otrsv1/aes256otrsv1_t-aes_enc_only.cpp \
# EACirc/projects/caesar/aead/aes256otrsv1/Aes256otrsv1.cpp \
# EACirc/projects/caesar/aead/aescopav1/aescopav1_aes-core.cpp \
# EACirc/projects/caesar/aead/aescopav1/aescopav1_encrypt.cpp \
# EACirc/projects/caesar/aead/aescopav1/Aescopav1.cpp \
# EACirc/projects/caesar/aead/aesjambuv1/aesjambuv1_aes.cpp \
# EACirc/projects/caesar/aead/aesjambuv1/aesjambuv1_encrypt.cpp \
# EACirc/projects/caesar/aead/aesjambuv1/Aesjambuv1.cpp \
# EACirc/projects/caesar/aead/aezv1/aezv1_aez_ref.cpp \
# EACirc/projects/caesar/aead/aezv1/aezv1_encrypt.cpp \
# EACirc/projects/caesar/aead/aezv1/aezv1_rijndael-alg-fst.cpp \
# EACirc/projects/caesar/aead/aezv1/Aezv1.cpp \
# EACirc/projects/caesar/aead/aezv3/aezv3_aez_ref.cpp \
# EACirc/projects/caesar/aead/aezv3/aezv3_rijndael-alg-fst.cpp \
# EACirc/projects/caesar/aead/aezv3/Aezv3.cpp \
# EACirc/projects/caesar/aead/artemia128v1/artemia128v1_encrypt.cpp \
# EACirc/projects/caesar/aead/artemia128v1/artemia128v1_jhae_decryption.cpp \
# EACirc/projects/caesar/aead/artemia128v1/artemia128v1_jhae_encryption.cpp \
# EACirc/projects/caesar/aead/artemia128v1/artemia128v1_jhae_padding.cpp \
# EACirc/projects/caesar/aead/artemia128v1/artemia128v1_jhae_permutation.cpp \
# EACirc/projects/caesar/aead/artemia128v1/Artemia128v1.cpp \
# EACirc/projects/caesar/aead/artemia256v1/artemia256v1_encrypt.cpp \
# EACirc/projects/caesar/aead/artemia256v1/artemia256v1_jhae_decryption.cpp \
# EACirc/projects/caesar/aead/artemia256v1/artemia256v1_jhae_encryption.cpp \
# EACirc/projects/caesar/aead/artemia256v1/artemia256v1_jhae_padding.cpp \
# EACirc/projects/caesar/aead/artemia256v1/artemia256v1_jhae_permutation.cpp \
# EACirc/projects/caesar/aead/artemia256v1/Artemia256v1.cpp \
# EACirc/projects/caesar/aead/ascon128v1/ascon128v1_ascon.cpp \
# EACirc/projects/caesar/aead/ascon128v1/Ascon128v1.cpp \
# EACirc/projects/caesar/aead/ascon96v1/ascon96v1_ascon.cpp \
# EACirc/projects/caesar/aead/ascon96v1/Ascon96v1.cpp \
# EACirc/projects/caesar/aead/calicov8/calicov8_calico.cpp \
# EACirc/projects/caesar/aead/calicov8/calicov8_encrypt.cpp \
# EACirc/projects/caesar/aead/calicov8/Calicov8.cpp \
# EACirc/projects/caesar/aead/cba1/cba1_encrypt.cpp \
# EACirc/projects/caesar/aead/cba1/Cba1.cpp \
# EACirc/projects/caesar/aead/cba10/cba10_encrypt.cpp \
# EACirc/projects/caesar/aead/cba10/Cba10.cpp \
# EACirc/projects/caesar/aead/cba2/cba2_encrypt.cpp \
# EACirc/projects/caesar/aead/cba2/Cba2.cpp \
# EACirc/projects/caesar/aead/cba3/cba3_encrypt.cpp \
# EACirc/projects/caesar/aead/cba3/Cba3.cpp \
# EACirc/projects/caesar/aead/cba4/cba4_encrypt.cpp \
# EACirc/projects/caesar/aead/cba4/Cba4.cpp \
# EACirc/projects/caesar/aead/cba5/cba5_encrypt.cpp \
# EACirc/projects/caesar/aead/cba5/Cba5.cpp \
# EACirc/projects/caesar/aead/cba6/cba6_encrypt.cpp \
# EACirc/projects/caesar/aead/cba6/Cba6.cpp \
# EACirc/projects/caesar/aead/cba7/cba7_encrypt.cpp \
# EACirc/projects/caesar/aead/cba7/Cba7.cpp \
# EACirc/projects/caesar/aead/cba8/cba8_encrypt.cpp \
# EACirc/projects/caesar/aead/cba8/Cba8.cpp \
# EACirc/projects/caesar/aead/cba9/cba9_encrypt.cpp \
# EACirc/projects/caesar/aead/cba9/Cba9.cpp \
# EACirc/projects/caesar/aead/cmcc22v1/cmcc22v1_encrypt.cpp \
# EACirc/projects/caesar/aead/cmcc22v1/Cmcc22v1.cpp \
# EACirc/projects/caesar/aead/cmcc24v1/cmcc24v1_encrypt.cpp \
# EACirc/projects/caesar/aead/cmcc24v1/Cmcc24v1.cpp \
# EACirc/projects/caesar/aead/cmcc42v1/cmcc42v1_encrypt.cpp \
# EACirc/projects/caesar/aead/cmcc42v1/Cmcc42v1.cpp \
# EACirc/projects/caesar/aead/cmcc44v1/cmcc44v1_encrypt.cpp \
# EACirc/projects/caesar/aead/cmcc44v1/Cmcc44v1.cpp \
# EACirc/projects/caesar/aead/cmcc84v1/cmcc84v1_encrypt.cpp \
# EACirc/projects/caesar/aead/cmcc84v1/Cmcc84v1.cpp \
# EACirc/projects/caesar/aead/deoxyseq128128v1/deoxyseq128128v1_deoxys.cpp \
# EACirc/projects/caesar/aead/deoxyseq128128v1/deoxyseq128128v1_encrypt.cpp \
# EACirc/projects/caesar/aead/deoxyseq128128v1/deoxyseq128128v1_tweakableBC.cpp \
# EACirc/projects/caesar/aead/deoxyseq128128v1/Deoxyseq128128v1.cpp \
# EACirc/projects/caesar/aead/deoxyseq256128v1/deoxyseq256128v1_deoxys.cpp \
# EACirc/projects/caesar/aead/deoxyseq256128v1/deoxyseq256128v1_encrypt.cpp \
# EACirc/projects/caesar/aead/deoxyseq256128v1/deoxyseq256128v1_tweakableBC.cpp \
# EACirc/projects/caesar/aead/deoxyseq256128v1/Deoxyseq256128v1.cpp \
# EACirc/projects/caesar/aead/deoxysneq128128v1/deoxysneq128128v1_deoxys.cpp \
# EACirc/projects/caesar/aead/deoxysneq128128v1/deoxysneq128128v1_encrypt.cpp \
# EACirc/projects/caesar/aead/deoxysneq128128v1/deoxysneq128128v1_tweakableBC.cpp \
# EACirc/projects/caesar/aead/deoxysneq128128v1/Deoxysneq128128v1.cpp \
# EACirc/projects/caesar/aead/deoxysneq256128v1/deoxysneq256128v1_deoxys.cpp \
# EACirc/projects/caesar/aead/deoxysneq256128v1/deoxysneq256128v1_encrypt.cpp \
# EACirc/projects/caesar/aead/deoxysneq256128v1/deoxysneq256128v1_tweakableBC.cpp \
# EACirc/projects/caesar/aead/deoxysneq256128v1/Deoxysneq256128v1.cpp \
# EACirc/projects/caesar/aead/elmd1000v1/elmd1000v1_encrypt.cpp \
# EACirc/projects/caesar/aead/elmd1000v1/Elmd1000v1.cpp \
# EACirc/projects/caesar/aead/elmd1001v1/elmd1001v1_encrypt.cpp \
# EACirc/projects/caesar/aead/elmd1001v1/Elmd1001v1.cpp \
# EACirc/projects/caesar/aead/elmd101270v1/elmd101270v1_encrypt.cpp \
# EACirc/projects/caesar/aead/elmd101270v1/Elmd101270v1.cpp \
# EACirc/projects/caesar/aead/elmd101271v1/elmd101271v1_encrypt.cpp \
# EACirc/projects/caesar/aead/elmd101271v1/Elmd101271v1.cpp \
# EACirc/projects/caesar/aead/elmd500v1/elmd500v1_encrypt.cpp \
# EACirc/projects/caesar/aead/elmd500v1/Elmd500v1.cpp \
# EACirc/projects/caesar/aead/elmd501v1/elmd501v1_encrypt.cpp \
# EACirc/projects/caesar/aead/elmd501v1/Elmd501v1.cpp \
# EACirc/projects/caesar/aead/elmd51270v1/elmd51270v1_encrypt.cpp \
# EACirc/projects/caesar/aead/elmd51270v1/Elmd51270v1.cpp \
# EACirc/projects/caesar/aead/elmd51271v1/elmd51271v1_encrypt.cpp \
# EACirc/projects/caesar/aead/elmd51271v1/Elmd51271v1.cpp \
# EACirc/projects/caesar/aead/enchilada128v1/enchilada128v1_aes.cpp \
# EACirc/projects/caesar/aead/enchilada128v1/enchilada128v1_auth.cpp \
# EACirc/projects/caesar/aead/enchilada128v1/enchilada128v1_chacha.cpp \
# EACirc/projects/caesar/aead/enchilada128v1/enchilada128v1_enchilada.cpp \
# EACirc/projects/caesar/aead/enchilada128v1/Enchilada128v1.cpp \
# EACirc/projects/caesar/aead/enchilada256v1/enchilada256v1_aescrypt.cpp \
# EACirc/projects/caesar/aead/enchilada256v1/enchilada256v1_aestab.cpp \
# EACirc/projects/caesar/aead/enchilada256v1/enchilada256v1_auth.cpp \
# EACirc/projects/caesar/aead/enchilada256v1/enchilada256v1_chacha.cpp \
# EACirc/projects/caesar/aead/enchilada256v1/enchilada256v1_enchilada256.cpp \
# EACirc/projects/caesar/aead/enchilada256v1/Enchilada256v1.cpp \
# EACirc/projects/caesar/aead/hs1sivhiv1/hs1sivhiv1_encrypt.cpp \
# EACirc/projects/caesar/aead/hs1sivhiv1/Hs1sivhiv1.cpp \
# EACirc/projects/caesar/aead/hs1sivlov1/hs1sivlov1_encrypt.cpp \
# EACirc/projects/caesar/aead/hs1sivlov1/Hs1sivlov1.cpp \
# EACirc/projects/caesar/aead/hs1sivv1/hs1sivv1_encrypt.cpp \
# EACirc/projects/caesar/aead/hs1sivv1/Hs1sivv1.cpp \
# EACirc/projects/caesar/aead/icepole128av1/icepole128av1_encrypt.cpp \
# EACirc/projects/caesar/aead/icepole128av1/icepole128av1_icepole.cpp \
# EACirc/projects/caesar/aead/icepole128av1/Icepole128av1.cpp \
# EACirc/projects/caesar/aead/icepole128v1/icepole128v1_encrypt.cpp \
# EACirc/projects/caesar/aead/icepole128v1/icepole128v1_icepole.cpp \
# EACirc/projects/caesar/aead/icepole128v1/Icepole128v1.cpp \
# EACirc/projects/caesar/aead/icepole256av1/icepole256av1_encrypt.cpp \
# EACirc/projects/caesar/aead/icepole256av1/icepole256av1_icepole.cpp \
# EACirc/projects/caesar/aead/icepole256av1/Icepole256av1.cpp \
# EACirc/projects/caesar/aead/ifeedaes128n104v1/ifeedaes128n104v1_encrypt.cpp \
# EACirc/projects/caesar/aead/ifeedaes128n104v1/Ifeedaes128n104v1.cpp \
# EACirc/projects/caesar/aead/ifeedaes128n96v1/ifeedaes128n96v1_encrypt.cpp \
# EACirc/projects/caesar/aead/ifeedaes128n96v1/Ifeedaes128n96v1.cpp \
# EACirc/projects/caesar/aead/iscream12v1/iscream12v1_iscream_cipher.cpp \
# EACirc/projects/caesar/aead/iscream12v1/iscream12v1_tae.cpp \
# EACirc/projects/caesar/aead/iscream12v1/Iscream12v1.cpp \
# EACirc/projects/caesar/aead/iscream12v2/iscream12v2_iscream_cipher.cpp \
# EACirc/projects/caesar/aead/iscream12v2/iscream12v2_tae.cpp \
# EACirc/projects/caesar/aead/iscream12v2/Iscream12v2.cpp \
# EACirc/projects/caesar/aead/iscream14v1/iscream14v1_iscream_cipher.cpp \
# EACirc/projects/caesar/aead/iscream14v1/iscream14v1_tae.cpp \
# EACirc/projects/caesar/aead/iscream14v1/Iscream14v1.cpp \
# EACirc/projects/caesar/aead/iscream14v2/iscream14v2_iscream_cipher.cpp \
# EACirc/projects/caesar/aead/iscream14v2/iscream14v2_tae.cpp \
# EACirc/projects/caesar/aead/iscream14v2/Iscream14v2.cpp \
# EACirc/projects/caesar/aead/joltikeq12864v1/joltikeq12864v1_encrypt.cpp \
# EACirc/projects/caesar/aead/joltikeq12864v1/joltikeq12864v1_joltik.cpp \
# EACirc/projects/caesar/aead/joltikeq12864v1/joltikeq12864v1_tweakableBC.cpp \
# EACirc/projects/caesar/aead/joltikeq12864v1/Joltikeq12864v1.cpp \
# EACirc/projects/caesar/aead/joltikeq6464v1/joltikeq6464v1_encrypt.cpp \
# EACirc/projects/caesar/aead/joltikeq6464v1/joltikeq6464v1_joltik.cpp \
# EACirc/projects/caesar/aead/joltikeq6464v1/joltikeq6464v1_tweakableBC.cpp \
# EACirc/projects/caesar/aead/joltikeq6464v1/Joltikeq6464v1.cpp \
# EACirc/projects/caesar/aead/joltikeq8048v1/joltikeq8048v1_encrypt.cpp \
# EACirc/projects/caesar/aead/joltikeq8048v1/joltikeq8048v1_joltik.cpp \
# EACirc/projects/caesar/aead/joltikeq8048v1/joltikeq8048v1_tweakableBC.cpp \
# EACirc/projects/caesar/aead/joltikeq8048v1/Joltikeq8048v1.cpp \
# EACirc/projects/caesar/aead/joltikeq9696v1/joltikeq9696v1_encrypt.cpp \
# EACirc/projects/caesar/aead/joltikeq9696v1/joltikeq9696v1_joltik.cpp \
# EACirc/projects/caesar/aead/joltikeq9696v1/joltikeq9696v1_tweakableBC.cpp \
# EACirc/projects/caesar/aead/joltikeq9696v1/Joltikeq9696v1.cpp \
# EACirc/projects/caesar/aead/joltikneq12864v1/joltikneq12864v1_encrypt.cpp \
# EACirc/projects/caesar/aead/joltikneq12864v1/joltikneq12864v1_joltik.cpp \
# EACirc/projects/caesar/aead/joltikneq12864v1/joltikneq12864v1_tweakableBC.cpp \
# EACirc/projects/caesar/aead/joltikneq12864v1/Joltikneq12864v1.cpp \
# EACirc/projects/caesar/aead/joltikneq6464v1/joltikneq6464v1_encrypt.cpp \
# EACirc/projects/caesar/aead/joltikneq6464v1/joltikneq6464v1_joltik.cpp \
# EACirc/projects/caesar/aead/joltikneq6464v1/joltikneq6464v1_tweakableBC.cpp \
# EACirc/projects/caesar/aead/joltikneq6464v1/Joltikneq6464v1.cpp \
# EACirc/projects/caesar/aead/joltikneq8048v1/joltikneq8048v1_encrypt.cpp \
# EACirc/projects/caesar/aead/joltikneq8048v1/joltikneq8048v1_joltik.cpp \
# EACirc/projects/caesar/aead/joltikneq8048v1/joltikneq8048v1_tweakableBC.cpp \
# EACirc/projects/caesar/aead/joltikneq8048v1/Joltikneq8048v1.cpp \
# EACirc/projects/caesar/aead/joltikneq9696v1/joltikneq9696v1_encrypt.cpp \
# EACirc/projects/caesar/aead/joltikneq9696v1/joltikneq9696v1_joltik.cpp \
# EACirc/projects/caesar/aead/joltikneq9696v1/joltikneq9696v1_tweakableBC.cpp \
# EACirc/projects/caesar/aead/joltikneq9696v1/Joltikneq9696v1.cpp \
# EACirc/projects/caesar/aead/juliusv1draft/juliusv1draft_aes.cpp \
# EACirc/projects/caesar/aead/juliusv1draft/juliusv1draft_encrypt.cpp \
# EACirc/projects/caesar/aead/juliusv1draft/juliusv1draft_functions.cpp \
# EACirc/projects/caesar/aead/juliusv1draft/Juliusv1draft.cpp \
# EACirc/projects/caesar/aead/ketjejrv1/ketjejrv1_displayIntermediateValues.cpp \
# EACirc/projects/caesar/aead/ketjejrv1/ketjejrv1_encrypt.cpp \
# EACirc/projects/caesar/aead/ketjejrv1/ketjejrv1_KeccakF-200-reference.cpp \
# EACirc/projects/caesar/aead/ketjejrv1/ketjejrv1_KeccakP-200-reference.cpp \
# EACirc/projects/caesar/aead/ketjejrv1/ketjejrv1_Ket.cpp \
# EACirc/projects/caesar/aead/ketjejrv1/ketjejrv1_Ketje.cpp \
# EACirc/projects/caesar/aead/ketjejrv1/Ketjejrv1.cpp \
# EACirc/projects/caesar/aead/ketjesrv1/ketjesrv1_displayIntermediateValues.cpp \
# EACirc/projects/caesar/aead/ketjesrv1/ketjesrv1_encrypt.cpp \
# EACirc/projects/caesar/aead/ketjesrv1/ketjesrv1_KeccakF-400-reference.cpp \
# EACirc/projects/caesar/aead/ketjesrv1/ketjesrv1_KeccakP-400-reference.cpp \
# EACirc/projects/caesar/aead/ketjesrv1/ketjesrv1_Ket.cpp \
# EACirc/projects/caesar/aead/ketjesrv1/ketjesrv1_Ketje.cpp \
# EACirc/projects/caesar/aead/ketjesrv1/Ketjesrv1.cpp \
# EACirc/projects/caesar/aead/kiasueq128v1/kiasueq128v1_encrypt.cpp \
# EACirc/projects/caesar/aead/kiasueq128v1/kiasueq128v1_kiasu.cpp \
# EACirc/projects/caesar/aead/kiasueq128v1/kiasueq128v1_tweakable_aes.cpp \
# EACirc/projects/caesar/aead/kiasueq128v1/Kiasueq128v1.cpp \
# EACirc/projects/caesar/aead/kiasuneq128v1/kiasuneq128v1_encrypt.cpp \
# EACirc/projects/caesar/aead/kiasuneq128v1/kiasuneq128v1_kiasu.cpp \
# EACirc/projects/caesar/aead/kiasuneq128v1/kiasuneq128v1_tweakable_aes.cpp \
# EACirc/projects/caesar/aead/kiasuneq128v1/Kiasuneq128v1.cpp \
# EACirc/projects/caesar/aead/lacv1/lacv1_encrypt.cpp \
# EACirc/projects/caesar/aead/lacv1/Lacv1.cpp \
# EACirc/projects/caesar/aead/lakekeyakv1/lakekeyakv1_displayIntermediateValues.cpp \
# EACirc/projects/caesar/aead/lakekeyakv1/lakekeyakv1_encrypt.cpp \
# EACirc/projects/caesar/aead/lakekeyakv1/lakekeyakv1_KeccakDuplex.cpp \
# EACirc/projects/caesar/aead/lakekeyakv1/lakekeyakv1_KeccakF-1600-reference.cpp \
# EACirc/projects/caesar/aead/lakekeyakv1/lakekeyakv1_KeccakP-1600-12-reference.cpp \
# EACirc/projects/caesar/aead/lakekeyakv1/lakekeyakv1_Keyak.cpp \
# EACirc/projects/caesar/aead/lakekeyakv1/Lakekeyakv1.cpp \
# EACirc/projects/caesar/aead/led80n6silcv1/led80n6silcv1_encrypt.cpp \
# EACirc/projects/caesar/aead/led80n6silcv1/led80n6silcv1_led.cpp \
# EACirc/projects/caesar/aead/led80n6silcv1/led80n6silcv1_silc.cpp \
# EACirc/projects/caesar/aead/led80n6silcv1/led80n6silcv1_utils.cpp \
# EACirc/projects/caesar/aead/led80n6silcv1/Led80n6silcv1.cpp \
# EACirc/projects/caesar/aead/minalpherv1/minalpherv1_encrypt.cpp \
# EACirc/projects/caesar/aead/minalpherv1/Minalpherv1.cpp \
# EACirc/projects/caesar/aead/morus1280128v1/morus1280128v1_encrypt.cpp \
# EACirc/projects/caesar/aead/morus1280128v1/Morus1280128v1.cpp \
# EACirc/projects/caesar/aead/morus1280256v1/morus1280256v1_encrypt.cpp \
# EACirc/projects/caesar/aead/morus1280256v1/Morus1280256v1.cpp \
# EACirc/projects/caesar/aead/morus640128v1/morus640128v1_encrypt.cpp \
# EACirc/projects/caesar/aead/morus640128v1/Morus640128v1.cpp \
# EACirc/projects/caesar/aead/norx3241v1/norx3241v1_caesar.cpp \
# EACirc/projects/caesar/aead/norx3241v1/norx3241v1_norx.cpp \
# EACirc/projects/caesar/aead/norx3241v1/Norx3241v1.cpp \
# EACirc/projects/caesar/aead/norx3261v1/norx3261v1_caesar.cpp \
# EACirc/projects/caesar/aead/norx3261v1/norx3261v1_norx.cpp \
# EACirc/projects/caesar/aead/norx3261v1/Norx3261v1.cpp \
# EACirc/projects/caesar/aead/norx6441v1/norx6441v1_caesar.cpp \
# EACirc/projects/caesar/aead/norx6441v1/norx6441v1_norx.cpp \
# EACirc/projects/caesar/aead/norx6441v1/Norx6441v1.cpp \
# EACirc/projects/caesar/aead/norx6444v1/norx6444v1_caesar.cpp \
# EACirc/projects/caesar/aead/norx6444v1/norx6444v1_norx.cpp \
# EACirc/projects/caesar/aead/norx6444v1/Norx6444v1.cpp \
# EACirc/projects/caesar/aead/norx6461v1/norx6461v1_caesar.cpp \
# EACirc/projects/caesar/aead/norx6461v1/norx6461v1_norx.cpp \
# EACirc/projects/caesar/aead/norx6461v1/Norx6461v1.cpp \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_displayIntermediateValues.cpp \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_encrypt.cpp \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_KeccakDuplex.cpp \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_KeccakF-1600-reference.cpp \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_KeccakP-1600-12-reference.cpp \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_KeccakParallelDuplex.cpp \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_Keyak.cpp \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_SerialFallback.cpp \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_testParallelPaSM.cpp \
# EACirc/projects/caesar/aead/oceankeyakv1/Oceankeyakv1.cpp \
# EACirc/projects/caesar/aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_encrypt.cpp \
# EACirc/projects/caesar/aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_omdsha256.cpp \
# EACirc/projects/caesar/aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_sha256.cpp \
# EACirc/projects/caesar/aead/omdsha256k128n96tau128v1/Omdsha256k128n96tau128v1.cpp \
# EACirc/projects/caesar/aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_encrypt.cpp \
# EACirc/projects/caesar/aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_omdsha256.cpp \
# EACirc/projects/caesar/aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_sha256.cpp \
# EACirc/projects/caesar/aead/omdsha256k128n96tau64v1/Omdsha256k128n96tau64v1.cpp \
# EACirc/projects/caesar/aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_encrypt.cpp \
# EACirc/projects/caesar/aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_omdsha256.cpp \
# EACirc/projects/caesar/aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_sha256.cpp \
# EACirc/projects/caesar/aead/omdsha256k128n96tau96v1/Omdsha256k128n96tau96v1.cpp \
# EACirc/projects/caesar/aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_encrypt.cpp \
# EACirc/projects/caesar/aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_omdsha256.cpp \
# EACirc/projects/caesar/aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_sha256.cpp \
# EACirc/projects/caesar/aead/omdsha256k192n104tau128v1/Omdsha256k192n104tau128v1.cpp \
# EACirc/projects/caesar/aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_encrypt.cpp \
# EACirc/projects/caesar/aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_omdsha256.cpp \
# EACirc/projects/caesar/aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_sha256.cpp \
# EACirc/projects/caesar/aead/omdsha256k256n104tau160v1/Omdsha256k256n104tau160v1.cpp \
# EACirc/projects/caesar/aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_encrypt.cpp \
# EACirc/projects/caesar/aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_omdsha256.cpp \
# EACirc/projects/caesar/aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_sha256.cpp \
# EACirc/projects/caesar/aead/omdsha256k256n248tau256v1/Omdsha256k256n248tau256v1.cpp \
# EACirc/projects/caesar/aead/omdsha512k128n128tau128v1/omdsha512k128n128tau128v1_encrypt.cpp \
# EACirc/projects/caesar/aead/omdsha512k128n128tau128v1/omdsha512k128n128tau128v1_omdsha512.cpp \
# EACirc/projects/caesar/aead/omdsha512k128n128tau128v1/omdsha512k128n128tau128v1_sha512.cpp \
# EACirc/projects/caesar/aead/omdsha512k128n128tau128v1/Omdsha512k128n128tau128v1.cpp \
# EACirc/projects/caesar/aead/omdsha512k256n256tau256v1/omdsha512k256n256tau256v1_encrypt.cpp \
# EACirc/projects/caesar/aead/omdsha512k256n256tau256v1/omdsha512k256n256tau256v1_omdsha512.cpp \
# EACirc/projects/caesar/aead/omdsha512k256n256tau256v1/omdsha512k256n256tau256v1_sha512.cpp \
# EACirc/projects/caesar/aead/omdsha512k256n256tau256v1/Omdsha512k256n256tau256v1.cpp \
# EACirc/projects/caesar/aead/omdsha512k512n256tau256v1/omdsha512k512n256tau256v1_encrypt.cpp \
# EACirc/projects/caesar/aead/omdsha512k512n256tau256v1/omdsha512k512n256tau256v1_omdsha512.cpp \
# EACirc/projects/caesar/aead/omdsha512k512n256tau256v1/omdsha512k512n256tau256v1_sha512.cpp \
# EACirc/projects/caesar/aead/omdsha512k512n256tau256v1/Omdsha512k512n256tau256v1.cpp \
# EACirc/projects/caesar/aead/paeq128/paeq128_encrypt.cpp \
# EACirc/projects/caesar/aead/paeq128/Paeq128.cpp \
# EACirc/projects/caesar/aead/paeq128t/paeq128t_encrypt.cpp \
# EACirc/projects/caesar/aead/paeq128t/Paeq128t.cpp \
# EACirc/projects/caesar/aead/paeq128tnm/paeq128tnm_encrypt.cpp \
# EACirc/projects/caesar/aead/paeq128tnm/Paeq128tnm.cpp \
# EACirc/projects/caesar/aead/paeq160/paeq160_encrypt.cpp \
# EACirc/projects/caesar/aead/paeq160/Paeq160.cpp \
# EACirc/projects/caesar/aead/paeq64/paeq64_encrypt.cpp \
# EACirc/projects/caesar/aead/paeq64/Paeq64.cpp \
# EACirc/projects/caesar/aead/paeq80/paeq80_encrypt.cpp \
# EACirc/projects/caesar/aead/paeq80/Paeq80.cpp \
# EACirc/projects/caesar/aead/pi16cipher096v1/pi16cipher096v1_encrypt.cpp \
# EACirc/projects/caesar/aead/pi16cipher096v1/Pi16cipher096v1.cpp \
# EACirc/projects/caesar/aead/pi16cipher128v1/pi16cipher128v1_encrypt.cpp \
# EACirc/projects/caesar/aead/pi16cipher128v1/Pi16cipher128v1.cpp \
# EACirc/projects/caesar/aead/pi32cipher128v1/pi32cipher128v1_encrypt.cpp \
# EACirc/projects/caesar/aead/pi32cipher128v1/Pi32cipher128v1.cpp \
# EACirc/projects/caesar/aead/pi32cipher256v1/pi32cipher256v1_encrypt.cpp \
# EACirc/projects/caesar/aead/pi32cipher256v1/Pi32cipher256v1.cpp \
# EACirc/projects/caesar/aead/pi64cipher128v1/pi64cipher128v1_encrypt.cpp \
# EACirc/projects/caesar/aead/pi64cipher128v1/Pi64cipher128v1.cpp \
# EACirc/projects/caesar/aead/pi64cipher256v1/pi64cipher256v1_encrypt.cpp \
# EACirc/projects/caesar/aead/pi64cipher256v1/Pi64cipher256v1.cpp \
# EACirc/projects/caesar/aead/pi64cipher256v1oneround/pi64cipher256v1oneround_encrypt.cpp \
# EACirc/projects/caesar/aead/pi64cipher256v1oneround/Pi64cipher256v1oneround.cpp \
# EACirc/projects/caesar/aead/pi64cipher256v1tworounds/pi64cipher256v1tworounds_encrypt.cpp \
# EACirc/projects/caesar/aead/pi64cipher256v1tworounds/Pi64cipher256v1tworounds.cpp \
# EACirc/projects/caesar/aead/polawisv1/polawisv1_decode_POLAWIS.cpp \
# EACirc/projects/caesar/aead/polawisv1/polawisv1_encode_POLAWIS.cpp \
# EACirc/projects/caesar/aead/polawisv1/polawisv1_key_gen_POLAWIS.cpp \
# EACirc/projects/caesar/aead/polawisv1/Polawisv1.cpp \
# EACirc/projects/caesar/aead/ppaev11/ppaev11_encrypt.cpp \
# EACirc/projects/caesar/aead/ppaev11/Ppaev11.cpp \
# EACirc/projects/caesar/aead/present80n6silcv1/present80n6silcv1_encrypt.cpp \
# EACirc/projects/caesar/aead/present80n6silcv1/present80n6silcv1_present.cpp \
# EACirc/projects/caesar/aead/present80n6silcv1/present80n6silcv1_silc.cpp \
# EACirc/projects/caesar/aead/present80n6silcv1/present80n6silcv1_utils.cpp \
# EACirc/projects/caesar/aead/present80n6silcv1/Present80n6silcv1.cpp \
# EACirc/projects/caesar/aead/primatesv1ape120/primatesv1ape120_encrypt.cpp \
# EACirc/projects/caesar/aead/primatesv1ape120/primatesv1ape120_primate.cpp \
# EACirc/projects/caesar/aead/primatesv1ape120/Primatesv1ape120.cpp \
# EACirc/projects/caesar/aead/primatesv1ape80/primatesv1ape80_encrypt.cpp \
# EACirc/projects/caesar/aead/primatesv1ape80/primatesv1ape80_primate.cpp \
# EACirc/projects/caesar/aead/primatesv1ape80/Primatesv1ape80.cpp \
# EACirc/projects/caesar/aead/primatesv1gibbon120/primatesv1gibbon120_encrypt.cpp \
# EACirc/projects/caesar/aead/primatesv1gibbon120/primatesv1gibbon120_primate.cpp \
# EACirc/projects/caesar/aead/primatesv1gibbon120/Primatesv1gibbon120.cpp \
# EACirc/projects/caesar/aead/primatesv1gibbon80/primatesv1gibbon80_encrypt.cpp \
# EACirc/projects/caesar/aead/primatesv1gibbon80/primatesv1gibbon80_primate.cpp \
# EACirc/projects/caesar/aead/primatesv1gibbon80/Primatesv1gibbon80.cpp \
# EACirc/projects/caesar/aead/primatesv1hanuman120/primatesv1hanuman120_encrypt.cpp \
# EACirc/projects/caesar/aead/primatesv1hanuman120/primatesv1hanuman120_primate.cpp \
# EACirc/projects/caesar/aead/primatesv1hanuman120/Primatesv1hanuman120.cpp \
# EACirc/projects/caesar/aead/primatesv1hanuman80/primatesv1hanuman80_encrypt.cpp \
# EACirc/projects/caesar/aead/primatesv1hanuman80/primatesv1hanuman80_primate.cpp \
# EACirc/projects/caesar/aead/primatesv1hanuman80/Primatesv1hanuman80.cpp \
# EACirc/projects/caesar/aead/proest128apev1/proest128apev1_encrypt.cpp \
# EACirc/projects/caesar/aead/proest128apev1/proest128apev1_proest128.cpp \
# EACirc/projects/caesar/aead/proest128apev1/proest128apev1_zerobytes.cpp \
# EACirc/projects/caesar/aead/proest128apev1/Proest128apev1.cpp \
# EACirc/projects/caesar/aead/proest128copav1/proest128copav1_encrypt.cpp \
# EACirc/projects/caesar/aead/proest128copav1/proest128copav1_proest128.cpp \
# EACirc/projects/caesar/aead/proest128copav1/proest128copav1_zerobytes.cpp \
# EACirc/projects/caesar/aead/proest128copav1/Proest128copav1.cpp \
# EACirc/projects/caesar/aead/proest128otrv1/proest128otrv1_encrypt.cpp \
# EACirc/projects/caesar/aead/proest128otrv1/proest128otrv1_proest128.cpp \
# EACirc/projects/caesar/aead/proest128otrv1/proest128otrv1_zerobytes.cpp \
# EACirc/projects/caesar/aead/proest128otrv1/Proest128otrv1.cpp \
# EACirc/projects/caesar/aead/proest256apev1/proest256apev1_encrypt.cpp \
# EACirc/projects/caesar/aead/proest256apev1/proest256apev1_proest256.cpp \
# EACirc/projects/caesar/aead/proest256apev1/proest256apev1_zerobytes.cpp \
# EACirc/projects/caesar/aead/proest256apev1/Proest256apev1.cpp \
# EACirc/projects/caesar/aead/proest256copav1/proest256copav1_encrypt.cpp \
# EACirc/projects/caesar/aead/proest256copav1/proest256copav1_proest256.cpp \
# EACirc/projects/caesar/aead/proest256copav1/proest256copav1_zerobytes.cpp \
# EACirc/projects/caesar/aead/proest256copav1/Proest256copav1.cpp \
# EACirc/projects/caesar/aead/proest256otrv1/proest256otrv1_encrypt.cpp \
# EACirc/projects/caesar/aead/proest256otrv1/proest256otrv1_proest256.cpp \
# EACirc/projects/caesar/aead/proest256otrv1/proest256otrv1_zerobytes.cpp \
# EACirc/projects/caesar/aead/proest256otrv1/Proest256otrv1.cpp \
# EACirc/projects/caesar/aead/raviyoylav1/raviyoylav1_encrypt.cpp \
# EACirc/projects/caesar/aead/raviyoylav1/Raviyoylav1.cpp \
# EACirc/projects/caesar/aead/riverkeyakv1/riverkeyakv1_displayIntermediateValues.cpp \
# EACirc/projects/caesar/aead/riverkeyakv1/riverkeyakv1_encrypt.cpp \
# EACirc/projects/caesar/aead/riverkeyakv1/riverkeyakv1_KeccakDuplex.cpp \
# EACirc/projects/caesar/aead/riverkeyakv1/riverkeyakv1_KeccakF-800-reference.cpp \
# EACirc/projects/caesar/aead/riverkeyakv1/riverkeyakv1_KeccakP-800-12-reference.cpp \
# EACirc/projects/caesar/aead/riverkeyakv1/riverkeyakv1_Keyak.cpp \
# EACirc/projects/caesar/aead/riverkeyakv1/Riverkeyakv1.cpp \
# EACirc/projects/caesar/aead/sablierv1/sablierv1_encrypt.cpp \
# EACirc/projects/caesar/aead/sablierv1/Sablierv1.cpp \
# EACirc/projects/caesar/aead/scream10v1/scream10v1_scream_cipher.cpp \
# EACirc/projects/caesar/aead/scream10v1/scream10v1_tae.cpp \
# EACirc/projects/caesar/aead/scream10v1/Scream10v1.cpp \
# EACirc/projects/caesar/aead/scream10v2/scream10v2_scream_cipher.cpp \
# EACirc/projects/caesar/aead/scream10v2/scream10v2_tae.cpp \
# EACirc/projects/caesar/aead/scream10v2/Scream10v2.cpp \
# EACirc/projects/caesar/aead/scream12v1/scream12v1_scream_cipher.cpp \
# EACirc/projects/caesar/aead/scream12v1/scream12v1_tae.cpp \
# EACirc/projects/caesar/aead/scream12v1/Scream12v1.cpp \
# EACirc/projects/caesar/aead/scream12v2/scream12v2_scream_cipher.cpp \
# EACirc/projects/caesar/aead/scream12v2/scream12v2_tae.cpp \
# EACirc/projects/caesar/aead/scream12v2/Scream12v2.cpp \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_displayIntermediateValues.cpp \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_encrypt.cpp \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_KeccakDuplex.cpp \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_KeccakF-1600-reference.cpp \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_KeccakP-1600-12-reference.cpp \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_KeccakParallelDuplex.cpp \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_Keyak.cpp \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_SerialFallback.cpp \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_testParallelPaSM.cpp \
# EACirc/projects/caesar/aead/seakeyakv1/Seakeyakv1.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d4n64/shellaes128v1d4n64_aes.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d4n64/shellaes128v1d4n64_aesReduced.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d4n64/shellaes128v1d4n64_encrypt.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d4n64/shellaes128v1d4n64_shellaes.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d4n64/Shellaes128v1d4n64.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d4n80/shellaes128v1d4n80_aes.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d4n80/shellaes128v1d4n80_aesReduced.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d4n80/shellaes128v1d4n80_encrypt.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d4n80/shellaes128v1d4n80_shellaes.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d4n80/Shellaes128v1d4n80.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d5n64/shellaes128v1d5n64_aes.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d5n64/shellaes128v1d5n64_aesReduced.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d5n64/shellaes128v1d5n64_encrypt.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d5n64/shellaes128v1d5n64_shellaes.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d5n64/Shellaes128v1d5n64.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d5n80/shellaes128v1d5n80_aes.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d5n80/shellaes128v1d5n80_aesReduced.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d5n80/shellaes128v1d5n80_encrypt.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d5n80/shellaes128v1d5n80_shellaes.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d5n80/Shellaes128v1d5n80.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d6n64/shellaes128v1d6n64_aes.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d6n64/shellaes128v1d6n64_aesReduced.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d6n64/shellaes128v1d6n64_encrypt.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d6n64/shellaes128v1d6n64_shellaes.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d6n64/Shellaes128v1d6n64.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d6n80/shellaes128v1d6n80_aes.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d6n80/shellaes128v1d6n80_aesReduced.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d6n80/shellaes128v1d6n80_encrypt.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d6n80/shellaes128v1d6n80_shellaes.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d6n80/Shellaes128v1d6n80.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d7n64/shellaes128v1d7n64_aes.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d7n64/shellaes128v1d7n64_aesReduced.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d7n64/shellaes128v1d7n64_encrypt.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d7n64/shellaes128v1d7n64_shellaes.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d7n64/Shellaes128v1d7n64.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d7n80/shellaes128v1d7n80_aes.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d7n80/shellaes128v1d7n80_aesReduced.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d7n80/shellaes128v1d7n80_encrypt.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d7n80/shellaes128v1d7n80_shellaes.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d7n80/Shellaes128v1d7n80.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d8n64/shellaes128v1d8n64_aes.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d8n64/shellaes128v1d8n64_aesReduced.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d8n64/shellaes128v1d8n64_encrypt.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d8n64/shellaes128v1d8n64_shellaes.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d8n64/Shellaes128v1d8n64.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d8n80/shellaes128v1d8n80_aes.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d8n80/shellaes128v1d8n80_aesReduced.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d8n80/shellaes128v1d8n80_encrypt.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d8n80/shellaes128v1d8n80_shellaes.cpp \
# EACirc/projects/caesar/aead/shellaes128v1d8n80/Shellaes128v1d8n80.cpp \
# EACirc/projects/caesar/aead/silverv1/silverv1_encrypt.cpp \
# EACirc/projects/caesar/aead/silverv1/silverv1_rijndaelEndianNeutral.cpp \
# EACirc/projects/caesar/aead/silverv1/Silverv1.cpp \
# EACirc/projects/caesar/aead/stribob192r1/stribob192r1_encrypt.cpp \
# EACirc/projects/caesar/aead/stribob192r1/stribob192r1_sbob_pi64.cpp \
# EACirc/projects/caesar/aead/stribob192r1/stribob192r1_sbob_tab64.cpp \
# EACirc/projects/caesar/aead/stribob192r1/stribob192r1_stribob.cpp \
# EACirc/projects/caesar/aead/stribob192r1/Stribob192r1.cpp \
# EACirc/projects/caesar/aead/tiaoxinv1/tiaoxinv1_aes_round.cpp \
# EACirc/projects/caesar/aead/tiaoxinv1/tiaoxinv1_encrypt.cpp \
# EACirc/projects/caesar/aead/tiaoxinv1/tiaoxinv1_tiaoxin-reference.cpp \
# EACirc/projects/caesar/aead/tiaoxinv1/Tiaoxinv1.cpp \
# EACirc/projects/caesar/aead/trivia0v1/trivia0v1_encrypt.cpp \
# EACirc/projects/caesar/aead/trivia0v1/Trivia0v1.cpp \
# EACirc/projects/caesar/aead/trivia128v1/trivia128v1_encrypt.cpp \
# EACirc/projects/caesar/aead/trivia128v1/Trivia128v1.cpp \
# EACirc/projects/caesar/aead/twine80n6clocv1/twine80n6clocv1_cloc.cpp \
# EACirc/projects/caesar/aead/twine80n6clocv1/twine80n6clocv1_encrypt.cpp \
# EACirc/projects/caesar/aead/twine80n6clocv1/twine80n6clocv1_twine.cpp \
# EACirc/projects/caesar/aead/twine80n6clocv1/twine80n6clocv1_utils.cpp \
# EACirc/projects/caesar/aead/twine80n6clocv1/Twine80n6clocv1.cpp \
# EACirc/projects/caesar/aead/wheeshtv1mr3fr1t128/wheeshtv1mr3fr1t128_encrypt.cpp \
# EACirc/projects/caesar/aead/wheeshtv1mr3fr1t128/Wheeshtv1mr3fr1t128.cpp \
# EACirc/projects/caesar/aead/wheeshtv1mr3fr1t256/wheeshtv1mr3fr1t256_encrypt.cpp \
# EACirc/projects/caesar/aead/wheeshtv1mr3fr1t256/Wheeshtv1mr3fr1t256.cpp \
# EACirc/projects/caesar/aead/wheeshtv1mr3fr3t256/wheeshtv1mr3fr3t256_encrypt.cpp \
# EACirc/projects/caesar/aead/wheeshtv1mr3fr3t256/Wheeshtv1mr3fr3t256.cpp \
# EACirc/projects/caesar/aead/wheeshtv1mr5fr7t256/wheeshtv1mr5fr7t256_encrypt.cpp \
# EACirc/projects/caesar/aead/wheeshtv1mr5fr7t256/Wheeshtv1mr5fr7t256.cpp \
# EACirc/projects/caesar/aead/yaes128v2/yaes128v2_aes-128.cpp \
# EACirc/projects/caesar/aead/yaes128v2/yaes128v2_encrypt.cpp \
# EACirc/projects/caesar/aead/yaes128v2/Yaes128v2.cpp \

HEADERS += \
 EACirc/projects/caesar/aead/acorn128/acorn128_api.h \
 EACirc/projects/caesar/aead/acorn128/acorn128_encrypt.h \
 EACirc/projects/caesar/aead/acorn128/Acorn128.h \
 EACirc/projects/caesar/aead/aeadaes128ocbtaglen128v1/aeadaes128ocbtaglen128v1_api.h \
 EACirc/projects/caesar/aead/aeadaes128ocbtaglen128v1/aeadaes128ocbtaglen128v1_encrypt.h \
 EACirc/projects/caesar/aead/aeadaes128ocbtaglen128v1/Aeadaes128ocbtaglen128v1.h \
 EACirc/projects/caesar/aead/aeadaes128ocbtaglen64v1/aeadaes128ocbtaglen64v1_api.h \
 EACirc/projects/caesar/aead/aeadaes128ocbtaglen64v1/aeadaes128ocbtaglen64v1_encrypt.h \
 EACirc/projects/caesar/aead/aeadaes128ocbtaglen64v1/Aeadaes128ocbtaglen64v1.h \
 EACirc/projects/caesar/aead/aeadaes128ocbtaglen96v1/aeadaes128ocbtaglen96v1_api.h \
 EACirc/projects/caesar/aead/aeadaes128ocbtaglen96v1/aeadaes128ocbtaglen96v1_encrypt.h \
 EACirc/projects/caesar/aead/aeadaes128ocbtaglen96v1/Aeadaes128ocbtaglen96v1.h \
 EACirc/projects/caesar/aead/aeadaes192ocbtaglen128v1/aeadaes192ocbtaglen128v1_api.h \
 EACirc/projects/caesar/aead/aeadaes192ocbtaglen128v1/aeadaes192ocbtaglen128v1_encrypt.h \
 EACirc/projects/caesar/aead/aeadaes192ocbtaglen128v1/Aeadaes192ocbtaglen128v1.h \
 EACirc/projects/caesar/aead/aeadaes192ocbtaglen64v1/aeadaes192ocbtaglen64v1_api.h \
 EACirc/projects/caesar/aead/aeadaes192ocbtaglen64v1/aeadaes192ocbtaglen64v1_encrypt.h \
 EACirc/projects/caesar/aead/aeadaes192ocbtaglen64v1/Aeadaes192ocbtaglen64v1.h \
 EACirc/projects/caesar/aead/aeadaes192ocbtaglen96v1/aeadaes192ocbtaglen96v1_api.h \
 EACirc/projects/caesar/aead/aeadaes192ocbtaglen96v1/aeadaes192ocbtaglen96v1_encrypt.h \
 EACirc/projects/caesar/aead/aeadaes192ocbtaglen96v1/Aeadaes192ocbtaglen96v1.h \
 EACirc/projects/caesar/aead/aeadaes256ocbtaglen128v1/aeadaes256ocbtaglen128v1_api.h \
 EACirc/projects/caesar/aead/aeadaes256ocbtaglen128v1/aeadaes256ocbtaglen128v1_encrypt.h \
 EACirc/projects/caesar/aead/aeadaes256ocbtaglen128v1/Aeadaes256ocbtaglen128v1.h \
 EACirc/projects/caesar/aead/aeadaes256ocbtaglen64v1/aeadaes256ocbtaglen64v1_api.h \
 EACirc/projects/caesar/aead/aeadaes256ocbtaglen64v1/aeadaes256ocbtaglen64v1_encrypt.h \
 EACirc/projects/caesar/aead/aeadaes256ocbtaglen64v1/Aeadaes256ocbtaglen64v1.h \
 EACirc/projects/caesar/aead/aeadaes256ocbtaglen96v1/aeadaes256ocbtaglen96v1_api.h \
 EACirc/projects/caesar/aead/aeadaes256ocbtaglen96v1/aeadaes256ocbtaglen96v1_encrypt.h \
 EACirc/projects/caesar/aead/aeadaes256ocbtaglen96v1/Aeadaes256ocbtaglen96v1.h \
 EACirc/projects/caesar/aead/aegis128/aegis128_api.h \
 EACirc/projects/caesar/aead/aegis128/aegis128_encrypt.h \
 EACirc/projects/caesar/aead/aegis128/Aegis128.h \
 EACirc/projects/caesar/aead/aegis128l/aegis128l_api.h \
 EACirc/projects/caesar/aead/aegis128l/aegis128l_encrypt.h \
 EACirc/projects/caesar/aead/aegis128l/Aegis128l.h \
 EACirc/projects/caesar/aead/aegis256/aegis256_api.h \
 EACirc/projects/caesar/aead/aegis256/aegis256_encrypt.h \
 EACirc/projects/caesar/aead/aegis256/Aegis256.h \
# EACirc/projects/caesar/aead/aes128avalanchev1/aes128avalanchev1_api.h \ # compilaation ok, sigsegv while running
# EACirc/projects/caesar/aead/aes128avalanchev1/aes128avalanchev1_avalanche.h \ # compilaation ok, sigsegv while running
# EACirc/projects/caesar/aead/aes128avalanchev1/aes128avalanchev1_encrypt.h \ # compilaation ok, sigsegv while running
# EACirc/projects/caesar/aead/aes128avalanchev1/Aes128avalanchev1.h \ # compilaation ok, sigsegv while running
# EACirc/projects/caesar/aead/aes128cpfbv1/aes128cpfbv1_api.h \
# EACirc/projects/caesar/aead/aes128cpfbv1/aes128cpfbv1_encrypt.h \
# EACirc/projects/caesar/aead/aes128cpfbv1/Aes128cpfbv1.h \

HEADERS += \
 EACirc/projects/caesar/aead/aes128gcmv1/aes128gcmv1_api.h \
 EACirc/projects/caesar/aead/aes128gcmv1/aes128gcmv1_encrypt.h \
 EACirc/projects/caesar/aead/aes128gcmv1/Aes128gcmv1.h \
 EACirc/projects/caesar/aead/aes128marble4rv1/aes128marble4rv1_aes.h \
 EACirc/projects/caesar/aead/aes128marble4rv1/aes128marble4rv1_aes_locl.h \
 EACirc/projects/caesar/aead/aes128marble4rv1/aes128marble4rv1_api.h \
 EACirc/projects/caesar/aead/aes128marble4rv1/aes128marble4rv1_marble.h \
 EACirc/projects/caesar/aead/aes128marble4rv1/aes128marble4rv1_encrypt.h \
 EACirc/projects/caesar/aead/aes128marble4rv1/Aes128marble4rv1.h \
 EACirc/projects/caesar/aead/aes128n12clocv1/aes128n12clocv1_aes.h \
 EACirc/projects/caesar/aead/aes128n12clocv1/aes128n12clocv1_aes_locl.h \
 EACirc/projects/caesar/aead/aes128n12clocv1/aes128n12clocv1_api.h \
 EACirc/projects/caesar/aead/aes128n12clocv1/aes128n12clocv1_cloc.h \
 EACirc/projects/caesar/aead/aes128n12clocv1/aes128n12clocv1_encrypt.h \
 EACirc/projects/caesar/aead/aes128n12clocv1/Aes128n12clocv1.h \
# EACirc/projects/caesar/aead/aes128n12silcv1/aes128n12silcv1_aes.h \
# EACirc/projects/caesar/aead/aes128n12silcv1/aes128n12silcv1_aes_locl.h \
# EACirc/projects/caesar/aead/aes128n12silcv1/aes128n12silcv1_api.h \
# EACirc/projects/caesar/aead/aes128n12silcv1/aes128n12silcv1_silc.h \
# EACirc/projects/caesar/aead/aes128n12silcv1/aes128n12silcv1_encrypt.h \
# EACirc/projects/caesar/aead/aes128n12silcv1/Aes128n12silcv1.h \

HEADERS += \
 EACirc/projects/caesar/aead/aes128n8clocv1/aes128n8clocv1_aes.h \
 EACirc/projects/caesar/aead/aes128n8clocv1/aes128n8clocv1_aes_locl.h \
 EACirc/projects/caesar/aead/aes128n8clocv1/aes128n8clocv1_api.h \
 EACirc/projects/caesar/aead/aes128n8clocv1/aes128n8clocv1_cloc.h \
 EACirc/projects/caesar/aead/aes128n8clocv1/aes128n8clocv1_encrypt.h \
 EACirc/projects/caesar/aead/aes128n8clocv1/Aes128n8clocv1.h \
# EACirc/projects/caesar/aead/aes128n8silcv1/aes128n8silcv1_aes.h \
# EACirc/projects/caesar/aead/aes128n8silcv1/aes128n8silcv1_aes_locl.h \
# EACirc/projects/caesar/aead/aes128n8silcv1/aes128n8silcv1_api.h \
# EACirc/projects/caesar/aead/aes128n8silcv1/aes128n8silcv1_silc.h \
# EACirc/projects/caesar/aead/aes128n8silcv1/aes128n8silcv1_encrypt.h \
# EACirc/projects/caesar/aead/aes128n8silcv1/Aes128n8silcv1.h \

HEADERS += \
 EACirc/projects/caesar/aead/aes128otrpv1/aes128otrpv1_api.h \
 EACirc/projects/caesar/aead/aes128otrpv1/aes128otrpv1_OTR.h \
 EACirc/projects/caesar/aead/aes128otrpv1/aes128otrpv1_t-aes_define.h \
 EACirc/projects/caesar/aead/aes128otrpv1/aes128otrpv1_t-aes_table_enc_only.h \
 EACirc/projects/caesar/aead/aes128otrpv1/aes128otrpv1_encrypt.h \
 EACirc/projects/caesar/aead/aes128otrpv1/Aes128otrpv1.h \
 EACirc/projects/caesar/aead/aes128otrsv1/aes128otrsv1_api.h \
 EACirc/projects/caesar/aead/aes128otrsv1/aes128otrsv1_OTR.h \
 EACirc/projects/caesar/aead/aes128otrsv1/aes128otrsv1_t-aes_define.h \
 EACirc/projects/caesar/aead/aes128otrsv1/aes128otrsv1_t-aes_table_enc_only.h \
 EACirc/projects/caesar/aead/aes128otrsv1/aes128otrsv1_encrypt.h \
 EACirc/projects/caesar/aead/aes128otrsv1/Aes128otrsv1.h \
 EACirc/projects/caesar/aead/aes128poetv1aes128/aes128poetv1aes128_aes.h \
 EACirc/projects/caesar/aead/aes128poetv1aes128/aes128poetv1aes128_api.h \
 EACirc/projects/caesar/aead/aes128poetv1aes128/aes128poetv1aes128_gf_mul.h \
 EACirc/projects/caesar/aead/aes128poetv1aes128/aes128poetv1aes128_poet.h \
 EACirc/projects/caesar/aead/aes128poetv1aes128/aes128poetv1aes128_encrypt.h \
 EACirc/projects/caesar/aead/aes128poetv1aes128/Aes128poetv1aes128.h \
 EACirc/projects/caesar/aead/aes128poetv1aes4/aes128poetv1aes4_aes.h \
 EACirc/projects/caesar/aead/aes128poetv1aes4/aes128poetv1aes4_api.h \
 EACirc/projects/caesar/aead/aes128poetv1aes4/aes128poetv1aes4_gf_mul.h \
 EACirc/projects/caesar/aead/aes128poetv1aes4/aes128poetv1aes4_poet.h \
 EACirc/projects/caesar/aead/aes128poetv1aes4/aes128poetv1aes4_encrypt.h \
 EACirc/projects/caesar/aead/aes128poetv1aes4/Aes128poetv1aes4.h \
 # EACirc/projects/caesar/aead/aes192avalanchev1/aes192avalanchev1_api.h \
 # EACirc/projects/caesar/aead/aes192avalanchev1/aes192avalanchev1_avalanche.h \
 # EACirc/projects/caesar/aead/aes192avalanchev1/aes192avalanchev1_encrypt.h \
 # EACirc/projects/caesar/aead/aes192avalanchev1/Aes192avalanchev1.h \
 # EACirc/projects/caesar/aead/aes256avalanchev1/aes256avalanchev1_api.h \
 # EACirc/projects/caesar/aead/aes256avalanchev1/aes256avalanchev1_avalanche.h \
 # EACirc/projects/caesar/aead/aes256avalanchev1/aes256avalanchev1_encrypt.h \
 # EACirc/projects/caesar/aead/aes256avalanchev1/Aes256avalanchev1.h \

HEADERS += \
# EACirc/projects/caesar/aead/aes256cpfbv1/aes256cpfbv1_api.h \
# EACirc/projects/caesar/aead/aes256cpfbv1/aes256cpfbv1_encrypt.h \
# EACirc/projects/caesar/aead/aes256cpfbv1/Aes256cpfbv1.h \
# EACirc/projects/caesar/aead/aes256gcmv1/aes256gcmv1_api.h \
# EACirc/projects/caesar/aead/aes256gcmv1/aes256gcmv1_encrypt.h \
# EACirc/projects/caesar/aead/aes256gcmv1/Aes256gcmv1.h \
# EACirc/projects/caesar/aead/aes256otrpv1/aes256otrpv1_api.h \
# EACirc/projects/caesar/aead/aes256otrpv1/aes256otrpv1_OTR.h \
# EACirc/projects/caesar/aead/aes256otrpv1/aes256otrpv1_t-aes_define.h \
# EACirc/projects/caesar/aead/aes256otrpv1/aes256otrpv1_t-aes_table_enc_only.h \
# EACirc/projects/caesar/aead/aes256otrpv1/aes256otrpv1_encrypt.h \
# EACirc/projects/caesar/aead/aes256otrpv1/Aes256otrpv1.h \
# EACirc/projects/caesar/aead/aes256otrsv1/aes256otrsv1_api.h \
# EACirc/projects/caesar/aead/aes256otrsv1/aes256otrsv1_OTR.h \
# EACirc/projects/caesar/aead/aes256otrsv1/aes256otrsv1_t-aes_define.h \
# EACirc/projects/caesar/aead/aes256otrsv1/aes256otrsv1_t-aes_table_enc_only.h \
# EACirc/projects/caesar/aead/aes256otrsv1/aes256otrsv1_encrypt.h \
# EACirc/projects/caesar/aead/aes256otrsv1/Aes256otrsv1.h \
# EACirc/projects/caesar/aead/aescopav1/aescopav1_aes-core.h \
# EACirc/projects/caesar/aead/aescopav1/aescopav1_api.h \
# EACirc/projects/caesar/aead/aescopav1/aescopav1_encrypt.h \
# EACirc/projects/caesar/aead/aescopav1/Aescopav1.h \
# EACirc/projects/caesar/aead/aesjambuv1/aesjambuv1_aes.h \
# EACirc/projects/caesar/aead/aesjambuv1/aesjambuv1_api.h \
# EACirc/projects/caesar/aead/aesjambuv1/aesjambuv1_encrypt.h \
# EACirc/projects/caesar/aead/aesjambuv1/Aesjambuv1.h \
# EACirc/projects/caesar/aead/aezv1/aezv1_api.h \
# EACirc/projects/caesar/aead/aezv1/aezv1_rijndael-alg-fst.h \
# EACirc/projects/caesar/aead/aezv1/aezv1_encrypt.h \
# EACirc/projects/caesar/aead/aezv1/Aezv1.h \
# EACirc/projects/caesar/aead/aezv3/aezv3_api.h \
# EACirc/projects/caesar/aead/aezv3/aezv3_rijndael-alg-fst.h \
# EACirc/projects/caesar/aead/aezv3/aezv3_encrypt.h \
# EACirc/projects/caesar/aead/aezv3/Aezv3.h \
# EACirc/projects/caesar/aead/artemia128v1/artemia128v1_api.h \
# EACirc/projects/caesar/aead/artemia128v1/artemia128v1_jhae.h \
# EACirc/projects/caesar/aead/artemia128v1/artemia128v1_encrypt.h \
# EACirc/projects/caesar/aead/artemia128v1/Artemia128v1.h \
# EACirc/projects/caesar/aead/artemia256v1/artemia256v1_api.h \
# EACirc/projects/caesar/aead/artemia256v1/artemia256v1_jhae.h \
# EACirc/projects/caesar/aead/artemia256v1/artemia256v1_encrypt.h \
# EACirc/projects/caesar/aead/artemia256v1/Artemia256v1.h \
# EACirc/projects/caesar/aead/ascon128v1/ascon128v1_api.h \
# EACirc/projects/caesar/aead/ascon128v1/ascon128v1_encrypt.h \
# EACirc/projects/caesar/aead/ascon128v1/Ascon128v1.h \
# EACirc/projects/caesar/aead/ascon96v1/ascon96v1_api.h \
# EACirc/projects/caesar/aead/ascon96v1/ascon96v1_encrypt.h \
# EACirc/projects/caesar/aead/ascon96v1/Ascon96v1.h \
# EACirc/projects/caesar/aead/calicov8/calicov8_api.h \
# EACirc/projects/caesar/aead/calicov8/calicov8_calico.h \
# EACirc/projects/caesar/aead/calicov8/calicov8_encrypt.h \
# EACirc/projects/caesar/aead/calicov8/Calicov8.h \
# EACirc/projects/caesar/aead/cba1/cba1_api.h \
# EACirc/projects/caesar/aead/cba1/cba1_encrypt.h \
# EACirc/projects/caesar/aead/cba1/Cba1.h \
# EACirc/projects/caesar/aead/cba10/cba10_api.h \
# EACirc/projects/caesar/aead/cba10/cba10_encrypt.h \
# EACirc/projects/caesar/aead/cba10/Cba10.h \
# EACirc/projects/caesar/aead/cba2/cba2_api.h \
# EACirc/projects/caesar/aead/cba2/cba2_encrypt.h \
# EACirc/projects/caesar/aead/cba2/Cba2.h \
# EACirc/projects/caesar/aead/cba3/cba3_api.h \
# EACirc/projects/caesar/aead/cba3/cba3_encrypt.h \
# EACirc/projects/caesar/aead/cba3/Cba3.h \
# EACirc/projects/caesar/aead/cba4/cba4_api.h \
# EACirc/projects/caesar/aead/cba4/cba4_encrypt.h \
# EACirc/projects/caesar/aead/cba4/Cba4.h \
# EACirc/projects/caesar/aead/cba5/cba5_api.h \
# EACirc/projects/caesar/aead/cba5/cba5_encrypt.h \
# EACirc/projects/caesar/aead/cba5/Cba5.h \
# EACirc/projects/caesar/aead/cba6/cba6_api.h \
# EACirc/projects/caesar/aead/cba6/cba6_encrypt.h \
# EACirc/projects/caesar/aead/cba6/Cba6.h \
# EACirc/projects/caesar/aead/cba7/cba7_api.h \
# EACirc/projects/caesar/aead/cba7/cba7_encrypt.h \
# EACirc/projects/caesar/aead/cba7/Cba7.h \
# EACirc/projects/caesar/aead/cba8/cba8_api.h \
# EACirc/projects/caesar/aead/cba8/cba8_encrypt.h \
# EACirc/projects/caesar/aead/cba8/Cba8.h \
# EACirc/projects/caesar/aead/cba9/cba9_api.h \
# EACirc/projects/caesar/aead/cba9/cba9_encrypt.h \
# EACirc/projects/caesar/aead/cba9/Cba9.h \
# EACirc/projects/caesar/aead/cmcc22v1/cmcc22v1_api.h \
# EACirc/projects/caesar/aead/cmcc22v1/cmcc22v1_encrypt.h \
# EACirc/projects/caesar/aead/cmcc22v1/Cmcc22v1.h \
# EACirc/projects/caesar/aead/cmcc24v1/cmcc24v1_api.h \
# EACirc/projects/caesar/aead/cmcc24v1/cmcc24v1_encrypt.h \
# EACirc/projects/caesar/aead/cmcc24v1/Cmcc24v1.h \
# EACirc/projects/caesar/aead/cmcc42v1/cmcc42v1_api.h \
# EACirc/projects/caesar/aead/cmcc42v1/cmcc42v1_encrypt.h \
# EACirc/projects/caesar/aead/cmcc42v1/Cmcc42v1.h \
# EACirc/projects/caesar/aead/cmcc44v1/cmcc44v1_api.h \
# EACirc/projects/caesar/aead/cmcc44v1/cmcc44v1_encrypt.h \
# EACirc/projects/caesar/aead/cmcc44v1/Cmcc44v1.h \
# EACirc/projects/caesar/aead/cmcc84v1/cmcc84v1_api.h \
# EACirc/projects/caesar/aead/cmcc84v1/cmcc84v1_encrypt.h \
# EACirc/projects/caesar/aead/cmcc84v1/Cmcc84v1.h \
# EACirc/projects/caesar/aead/deoxyseq128128v1/deoxyseq128128v1_api.h \
# EACirc/projects/caesar/aead/deoxyseq128128v1/deoxyseq128128v1_crypto_aead.h \
# EACirc/projects/caesar/aead/deoxyseq128128v1/deoxyseq128128v1_deoxys.h \
# EACirc/projects/caesar/aead/deoxyseq128128v1/deoxyseq128128v1_tweakableBC.h \
# EACirc/projects/caesar/aead/deoxyseq128128v1/deoxyseq128128v1_encrypt.h \
# EACirc/projects/caesar/aead/deoxyseq128128v1/Deoxyseq128128v1.h \
# EACirc/projects/caesar/aead/deoxyseq256128v1/deoxyseq256128v1_api.h \
# EACirc/projects/caesar/aead/deoxyseq256128v1/deoxyseq256128v1_crypto_aead.h \
# EACirc/projects/caesar/aead/deoxyseq256128v1/deoxyseq256128v1_deoxys.h \
# EACirc/projects/caesar/aead/deoxyseq256128v1/deoxyseq256128v1_tweakableBC.h \
# EACirc/projects/caesar/aead/deoxyseq256128v1/deoxyseq256128v1_encrypt.h \
# EACirc/projects/caesar/aead/deoxyseq256128v1/Deoxyseq256128v1.h \
# EACirc/projects/caesar/aead/deoxysneq128128v1/deoxysneq128128v1_api.h \
# EACirc/projects/caesar/aead/deoxysneq128128v1/deoxysneq128128v1_crypto_aead.h \
# EACirc/projects/caesar/aead/deoxysneq128128v1/deoxysneq128128v1_deoxys.h \
# EACirc/projects/caesar/aead/deoxysneq128128v1/deoxysneq128128v1_tweakableBC.h \
# EACirc/projects/caesar/aead/deoxysneq128128v1/deoxysneq128128v1_encrypt.h \
# EACirc/projects/caesar/aead/deoxysneq128128v1/Deoxysneq128128v1.h \
# EACirc/projects/caesar/aead/deoxysneq256128v1/deoxysneq256128v1_api.h \
# EACirc/projects/caesar/aead/deoxysneq256128v1/deoxysneq256128v1_crypto_aead.h \
# EACirc/projects/caesar/aead/deoxysneq256128v1/deoxysneq256128v1_deoxys.h \
# EACirc/projects/caesar/aead/deoxysneq256128v1/deoxysneq256128v1_tweakableBC.h \
# EACirc/projects/caesar/aead/deoxysneq256128v1/deoxysneq256128v1_encrypt.h \
# EACirc/projects/caesar/aead/deoxysneq256128v1/Deoxysneq256128v1.h \
# EACirc/projects/caesar/aead/elmd1000v1/elmd1000v1_aes_round_5.h \
# EACirc/projects/caesar/aead/elmd1000v1/elmd1000v1_api.h \
# EACirc/projects/caesar/aead/elmd1000v1/elmd1000v1_module.h \
# EACirc/projects/caesar/aead/elmd1000v1/elmd1000v1_encrypt.h \
# EACirc/projects/caesar/aead/elmd1000v1/Elmd1000v1.h \
# EACirc/projects/caesar/aead/elmd1001v1/elmd1001v1_aes_round_5.h \
# EACirc/projects/caesar/aead/elmd1001v1/elmd1001v1_api.h \
# EACirc/projects/caesar/aead/elmd1001v1/elmd1001v1_module.h \
# EACirc/projects/caesar/aead/elmd1001v1/elmd1001v1_encrypt.h \
# EACirc/projects/caesar/aead/elmd1001v1/Elmd1001v1.h \
# EACirc/projects/caesar/aead/elmd101270v1/elmd101270v1_aes_round_5.h \
# EACirc/projects/caesar/aead/elmd101270v1/elmd101270v1_api.h \
# EACirc/projects/caesar/aead/elmd101270v1/elmd101270v1_module.h \
# EACirc/projects/caesar/aead/elmd101270v1/elmd101270v1_encrypt.h \
# EACirc/projects/caesar/aead/elmd101270v1/Elmd101270v1.h \
# EACirc/projects/caesar/aead/elmd101271v1/elmd101271v1_aes_round_5.h \
# EACirc/projects/caesar/aead/elmd101271v1/elmd101271v1_api.h \
# EACirc/projects/caesar/aead/elmd101271v1/elmd101271v1_module.h \
# EACirc/projects/caesar/aead/elmd101271v1/elmd101271v1_encrypt.h \
# EACirc/projects/caesar/aead/elmd101271v1/Elmd101271v1.h \
# EACirc/projects/caesar/aead/elmd500v1/elmd500v1_aes_round_5.h \
# EACirc/projects/caesar/aead/elmd500v1/elmd500v1_api.h \
# EACirc/projects/caesar/aead/elmd500v1/elmd500v1_module.h \
# EACirc/projects/caesar/aead/elmd500v1/elmd500v1_encrypt.h \
# EACirc/projects/caesar/aead/elmd500v1/Elmd500v1.h \
# EACirc/projects/caesar/aead/elmd501v1/elmd501v1_aes_round_5.h \
# EACirc/projects/caesar/aead/elmd501v1/elmd501v1_api.h \
# EACirc/projects/caesar/aead/elmd501v1/elmd501v1_module.h \
# EACirc/projects/caesar/aead/elmd501v1/elmd501v1_encrypt.h \
# EACirc/projects/caesar/aead/elmd501v1/Elmd501v1.h \
# EACirc/projects/caesar/aead/elmd51270v1/elmd51270v1_aes_round_5.h \
# EACirc/projects/caesar/aead/elmd51270v1/elmd51270v1_api.h \
# EACirc/projects/caesar/aead/elmd51270v1/elmd51270v1_module.h \
# EACirc/projects/caesar/aead/elmd51270v1/elmd51270v1_encrypt.h \
# EACirc/projects/caesar/aead/elmd51270v1/Elmd51270v1.h \
# EACirc/projects/caesar/aead/elmd51271v1/elmd51271v1_aes_round_5.h \
# EACirc/projects/caesar/aead/elmd51271v1/elmd51271v1_api.h \
# EACirc/projects/caesar/aead/elmd51271v1/elmd51271v1_module.h \
# EACirc/projects/caesar/aead/elmd51271v1/elmd51271v1_encrypt.h \
# EACirc/projects/caesar/aead/elmd51271v1/Elmd51271v1.h \
# EACirc/projects/caesar/aead/enchilada128v1/enchilada128v1_api.h \
# EACirc/projects/caesar/aead/enchilada128v1/enchilada128v1_enchilada.h \
# EACirc/projects/caesar/aead/enchilada128v1/enchilada128v1_encrypt.h \
# EACirc/projects/caesar/aead/enchilada128v1/Enchilada128v1.h \
# EACirc/projects/caesar/aead/enchilada256v1/enchilada256v1_aes.h \
# EACirc/projects/caesar/aead/enchilada256v1/enchilada256v1_aesopt.h \
# EACirc/projects/caesar/aead/enchilada256v1/enchilada256v1_api.h \
# EACirc/projects/caesar/aead/enchilada256v1/enchilada256v1_enchilada.h \
# EACirc/projects/caesar/aead/enchilada256v1/enchilada256v1_encrypt.h \
# EACirc/projects/caesar/aead/enchilada256v1/Enchilada256v1.h \
# EACirc/projects/caesar/aead/hs1sivhiv1/hs1sivhiv1_api.h \
# EACirc/projects/caesar/aead/hs1sivhiv1/hs1sivhiv1_encrypt.h \
# EACirc/projects/caesar/aead/hs1sivhiv1/Hs1sivhiv1.h \
# EACirc/projects/caesar/aead/hs1sivlov1/hs1sivlov1_api.h \
# EACirc/projects/caesar/aead/hs1sivlov1/hs1sivlov1_encrypt.h \
# EACirc/projects/caesar/aead/hs1sivlov1/Hs1sivlov1.h \
# EACirc/projects/caesar/aead/hs1sivv1/hs1sivv1_api.h \
# EACirc/projects/caesar/aead/hs1sivv1/hs1sivv1_encrypt.h \
# EACirc/projects/caesar/aead/hs1sivv1/Hs1sivv1.h \
# EACirc/projects/caesar/aead/icepole128av1/icepole128av1_api.h \
# EACirc/projects/caesar/aead/icepole128av1/icepole128av1_icepole.h \
# EACirc/projects/caesar/aead/icepole128av1/icepole128av1_encrypt.h \
# EACirc/projects/caesar/aead/icepole128av1/Icepole128av1.h \
# EACirc/projects/caesar/aead/icepole128v1/icepole128v1_api.h \
# EACirc/projects/caesar/aead/icepole128v1/icepole128v1_icepole.h \
# EACirc/projects/caesar/aead/icepole128v1/icepole128v1_encrypt.h \
# EACirc/projects/caesar/aead/icepole128v1/Icepole128v1.h \
# EACirc/projects/caesar/aead/icepole256av1/icepole256av1_api.h \
# EACirc/projects/caesar/aead/icepole256av1/icepole256av1_icepole.h \
# EACirc/projects/caesar/aead/icepole256av1/icepole256av1_encrypt.h \
# EACirc/projects/caesar/aead/icepole256av1/Icepole256av1.h \
# EACirc/projects/caesar/aead/ifeedaes128n104v1/ifeedaes128n104v1_api.h \
# EACirc/projects/caesar/aead/ifeedaes128n104v1/ifeedaes128n104v1_encrypt.h \
# EACirc/projects/caesar/aead/ifeedaes128n104v1/Ifeedaes128n104v1.h \
# EACirc/projects/caesar/aead/ifeedaes128n96v1/ifeedaes128n96v1_api.h \
# EACirc/projects/caesar/aead/ifeedaes128n96v1/ifeedaes128n96v1_encrypt.h \
# EACirc/projects/caesar/aead/ifeedaes128n96v1/Ifeedaes128n96v1.h \
# EACirc/projects/caesar/aead/iscream12v1/iscream12v1_api.h \
# EACirc/projects/caesar/aead/iscream12v1/iscream12v1_lbox.h \
# EACirc/projects/caesar/aead/iscream12v1/iscream12v1_params.h \
# EACirc/projects/caesar/aead/iscream12v1/iscream12v1_sbox.h \
# EACirc/projects/caesar/aead/iscream12v1/iscream12v1_encrypt.h \
# EACirc/projects/caesar/aead/iscream12v1/Iscream12v1.h \
# EACirc/projects/caesar/aead/iscream12v2/iscream12v2_api.h \
# EACirc/projects/caesar/aead/iscream12v2/iscream12v2_lbox.h \
# EACirc/projects/caesar/aead/iscream12v2/iscream12v2_params.h \
# EACirc/projects/caesar/aead/iscream12v2/iscream12v2_sbox.h \
# EACirc/projects/caesar/aead/iscream12v2/iscream12v2_encrypt.h \
# EACirc/projects/caesar/aead/iscream12v2/Iscream12v2.h \
# EACirc/projects/caesar/aead/iscream14v1/iscream14v1_api.h \
# EACirc/projects/caesar/aead/iscream14v1/iscream14v1_lbox.h \
# EACirc/projects/caesar/aead/iscream14v1/iscream14v1_params.h \
# EACirc/projects/caesar/aead/iscream14v1/iscream14v1_sbox.h \
# EACirc/projects/caesar/aead/iscream14v1/iscream14v1_encrypt.h \
# EACirc/projects/caesar/aead/iscream14v1/Iscream14v1.h \
# EACirc/projects/caesar/aead/iscream14v2/iscream14v2_api.h \
# EACirc/projects/caesar/aead/iscream14v2/iscream14v2_lbox.h \
# EACirc/projects/caesar/aead/iscream14v2/iscream14v2_params.h \
# EACirc/projects/caesar/aead/iscream14v2/iscream14v2_sbox.h \
# EACirc/projects/caesar/aead/iscream14v2/iscream14v2_encrypt.h \
# EACirc/projects/caesar/aead/iscream14v2/Iscream14v2.h \
# EACirc/projects/caesar/aead/joltikeq12864v1/joltikeq12864v1_api.h \
# EACirc/projects/caesar/aead/joltikeq12864v1/joltikeq12864v1_crypto_aead.h \
# EACirc/projects/caesar/aead/joltikeq12864v1/joltikeq12864v1_joltik.h \
# EACirc/projects/caesar/aead/joltikeq12864v1/joltikeq12864v1_tweakableBC.h \
# EACirc/projects/caesar/aead/joltikeq12864v1/joltikeq12864v1_encrypt.h \
# EACirc/projects/caesar/aead/joltikeq12864v1/Joltikeq12864v1.h \
# EACirc/projects/caesar/aead/joltikeq6464v1/joltikeq6464v1_api.h \
# EACirc/projects/caesar/aead/joltikeq6464v1/joltikeq6464v1_crypto_aead.h \
# EACirc/projects/caesar/aead/joltikeq6464v1/joltikeq6464v1_joltik.h \
# EACirc/projects/caesar/aead/joltikeq6464v1/joltikeq6464v1_tweakableBC.h \
# EACirc/projects/caesar/aead/joltikeq6464v1/joltikeq6464v1_encrypt.h \
# EACirc/projects/caesar/aead/joltikeq6464v1/Joltikeq6464v1.h \
# EACirc/projects/caesar/aead/joltikeq8048v1/joltikeq8048v1_api.h \
# EACirc/projects/caesar/aead/joltikeq8048v1/joltikeq8048v1_crypto_aead.h \
# EACirc/projects/caesar/aead/joltikeq8048v1/joltikeq8048v1_joltik.h \
# EACirc/projects/caesar/aead/joltikeq8048v1/joltikeq8048v1_tweakableBC.h \
# EACirc/projects/caesar/aead/joltikeq8048v1/joltikeq8048v1_encrypt.h \
# EACirc/projects/caesar/aead/joltikeq8048v1/Joltikeq8048v1.h \
# EACirc/projects/caesar/aead/joltikeq9696v1/joltikeq9696v1_api.h \
# EACirc/projects/caesar/aead/joltikeq9696v1/joltikeq9696v1_crypto_aead.h \
# EACirc/projects/caesar/aead/joltikeq9696v1/joltikeq9696v1_joltik.h \
# EACirc/projects/caesar/aead/joltikeq9696v1/joltikeq9696v1_tweakableBC.h \
# EACirc/projects/caesar/aead/joltikeq9696v1/joltikeq9696v1_encrypt.h \
# EACirc/projects/caesar/aead/joltikeq9696v1/Joltikeq9696v1.h \
# EACirc/projects/caesar/aead/joltikneq12864v1/joltikneq12864v1_api.h \
# EACirc/projects/caesar/aead/joltikneq12864v1/joltikneq12864v1_crypto_aead.h \
# EACirc/projects/caesar/aead/joltikneq12864v1/joltikneq12864v1_joltik.h \
# EACirc/projects/caesar/aead/joltikneq12864v1/joltikneq12864v1_tweakableBC.h \
# EACirc/projects/caesar/aead/joltikneq12864v1/joltikneq12864v1_encrypt.h \
# EACirc/projects/caesar/aead/joltikneq12864v1/Joltikneq12864v1.h \
# EACirc/projects/caesar/aead/joltikneq6464v1/joltikneq6464v1_api.h \
# EACirc/projects/caesar/aead/joltikneq6464v1/joltikneq6464v1_crypto_aead.h \
# EACirc/projects/caesar/aead/joltikneq6464v1/joltikneq6464v1_joltik.h \
# EACirc/projects/caesar/aead/joltikneq6464v1/joltikneq6464v1_tweakableBC.h \
# EACirc/projects/caesar/aead/joltikneq6464v1/joltikneq6464v1_encrypt.h \
# EACirc/projects/caesar/aead/joltikneq6464v1/Joltikneq6464v1.h \
# EACirc/projects/caesar/aead/joltikneq8048v1/joltikneq8048v1_api.h \
# EACirc/projects/caesar/aead/joltikneq8048v1/joltikneq8048v1_crypto_aead.h \
# EACirc/projects/caesar/aead/joltikneq8048v1/joltikneq8048v1_joltik.h \
# EACirc/projects/caesar/aead/joltikneq8048v1/joltikneq8048v1_tweakableBC.h \
# EACirc/projects/caesar/aead/joltikneq8048v1/joltikneq8048v1_encrypt.h \
# EACirc/projects/caesar/aead/joltikneq8048v1/Joltikneq8048v1.h \
# EACirc/projects/caesar/aead/joltikneq9696v1/joltikneq9696v1_api.h \
# EACirc/projects/caesar/aead/joltikneq9696v1/joltikneq9696v1_crypto_aead.h \
# EACirc/projects/caesar/aead/joltikneq9696v1/joltikneq9696v1_joltik.h \
# EACirc/projects/caesar/aead/joltikneq9696v1/joltikneq9696v1_tweakableBC.h \
# EACirc/projects/caesar/aead/joltikneq9696v1/joltikneq9696v1_encrypt.h \
# EACirc/projects/caesar/aead/joltikneq9696v1/Joltikneq9696v1.h \
# EACirc/projects/caesar/aead/juliusv1draft/juliusv1draft_api.h \
# EACirc/projects/caesar/aead/juliusv1draft/juliusv1draft_encrypt.h \
# EACirc/projects/caesar/aead/juliusv1draft/Juliusv1draft.h \
# EACirc/projects/caesar/aead/ketjejrv1/ketjejrv1_api.h \
# EACirc/projects/caesar/aead/ketjejrv1/ketjejrv1_brg_endian.h \
# EACirc/projects/caesar/aead/ketjejrv1/ketjejrv1_displayIntermediateValues.h \
# EACirc/projects/caesar/aead/ketjejrv1/ketjejrv1_KeccakF-200-interface.h \
# EACirc/projects/caesar/aead/ketjejrv1/ketjejrv1_KeccakF-200-reference.h \
# EACirc/projects/caesar/aead/ketjejrv1/ketjejrv1_KeccakF-interface.h \
# EACirc/projects/caesar/aead/ketjejrv1/ketjejrv1_KeccakP-200-interface.h \
# EACirc/projects/caesar/aead/ketjejrv1/ketjejrv1_KeccakP-200-reference.h \
# EACirc/projects/caesar/aead/ketjejrv1/ketjejrv1_KeccakP-interface.h \
# EACirc/projects/caesar/aead/ketjejrv1/ketjejrv1_Ket.h \
# EACirc/projects/caesar/aead/ketjejrv1/ketjejrv1_Ketje.h \
# EACirc/projects/caesar/aead/ketjejrv1/ketjejrv1_encrypt.h \
# EACirc/projects/caesar/aead/ketjejrv1/Ketjejrv1.h \
# EACirc/projects/caesar/aead/ketjesrv1/ketjesrv1_api.h \
# EACirc/projects/caesar/aead/ketjesrv1/ketjesrv1_brg_endian.h \
# EACirc/projects/caesar/aead/ketjesrv1/ketjesrv1_displayIntermediateValues.h \
# EACirc/projects/caesar/aead/ketjesrv1/ketjesrv1_KeccakF-400-interface.h \
# EACirc/projects/caesar/aead/ketjesrv1/ketjesrv1_KeccakF-400-reference.h \
# EACirc/projects/caesar/aead/ketjesrv1/ketjesrv1_KeccakF-interface.h \
# EACirc/projects/caesar/aead/ketjesrv1/ketjesrv1_KeccakP-400-interface.h \
# EACirc/projects/caesar/aead/ketjesrv1/ketjesrv1_KeccakP-400-reference.h \
# EACirc/projects/caesar/aead/ketjesrv1/ketjesrv1_KeccakP-interface.h \
# EACirc/projects/caesar/aead/ketjesrv1/ketjesrv1_Ket.h \
# EACirc/projects/caesar/aead/ketjesrv1/ketjesrv1_Ketje.h \
# EACirc/projects/caesar/aead/ketjesrv1/ketjesrv1_encrypt.h \
# EACirc/projects/caesar/aead/ketjesrv1/Ketjesrv1.h \
# EACirc/projects/caesar/aead/kiasueq128v1/kiasueq128v1_api.h \
# EACirc/projects/caesar/aead/kiasueq128v1/kiasueq128v1_crypto_aead.h \
# EACirc/projects/caesar/aead/kiasueq128v1/kiasueq128v1_kiasu.h \
# EACirc/projects/caesar/aead/kiasueq128v1/kiasueq128v1_tweakable_aes.h \
# EACirc/projects/caesar/aead/kiasueq128v1/kiasueq128v1_encrypt.h \
# EACirc/projects/caesar/aead/kiasueq128v1/Kiasueq128v1.h \
# EACirc/projects/caesar/aead/kiasuneq128v1/kiasuneq128v1_api.h \
# EACirc/projects/caesar/aead/kiasuneq128v1/kiasuneq128v1_kiasu.h \
# EACirc/projects/caesar/aead/kiasuneq128v1/kiasuneq128v1_tweakable_aes.h \
# EACirc/projects/caesar/aead/kiasuneq128v1/kiasuneq128v1_encrypt.h \
# EACirc/projects/caesar/aead/kiasuneq128v1/Kiasuneq128v1.h \
# EACirc/projects/caesar/aead/lacv1/lacv1_api.h \
# EACirc/projects/caesar/aead/lacv1/lacv1_encrypt.h \
# EACirc/projects/caesar/aead/lacv1/Lacv1.h \
# EACirc/projects/caesar/aead/lakekeyakv1/lakekeyakv1_api.h \
# EACirc/projects/caesar/aead/lakekeyakv1/lakekeyakv1_brg_endian.h \
# EACirc/projects/caesar/aead/lakekeyakv1/lakekeyakv1_displayIntermediateValues.h \
# EACirc/projects/caesar/aead/lakekeyakv1/lakekeyakv1_KeccakDuplex.h \
# EACirc/projects/caesar/aead/lakekeyakv1/lakekeyakv1_KeccakF-1600-interface.h \
# EACirc/projects/caesar/aead/lakekeyakv1/lakekeyakv1_KeccakF-1600-reference.h \
# EACirc/projects/caesar/aead/lakekeyakv1/lakekeyakv1_KeccakF-interface.h \
# EACirc/projects/caesar/aead/lakekeyakv1/lakekeyakv1_KeccakP-1600-12-interface.h \
# EACirc/projects/caesar/aead/lakekeyakv1/lakekeyakv1_KeccakP-1600-12-reference.h \
# EACirc/projects/caesar/aead/lakekeyakv1/lakekeyakv1_Keyak.h \
# EACirc/projects/caesar/aead/lakekeyakv1/lakekeyakv1_NumberOfParallelInstances.h \
# EACirc/projects/caesar/aead/lakekeyakv1/lakekeyakv1_encrypt.h \
# EACirc/projects/caesar/aead/lakekeyakv1/Lakekeyakv1.h \
# EACirc/projects/caesar/aead/led80n6silcv1/led80n6silcv1_api.h \
# EACirc/projects/caesar/aead/led80n6silcv1/led80n6silcv1_led.h \
# EACirc/projects/caesar/aead/led80n6silcv1/led80n6silcv1_silc.h \
# EACirc/projects/caesar/aead/led80n6silcv1/led80n6silcv1_encrypt.h \
# EACirc/projects/caesar/aead/led80n6silcv1/Led80n6silcv1.h \
# EACirc/projects/caesar/aead/minalpherv1/minalpherv1_api.h \
# EACirc/projects/caesar/aead/minalpherv1/minalpherv1_encrypt.h \
# EACirc/projects/caesar/aead/minalpherv1/Minalpherv1.h \
# EACirc/projects/caesar/aead/morus1280128v1/morus1280128v1_api.h \
# EACirc/projects/caesar/aead/morus1280128v1/morus1280128v1_encrypt.h \
# EACirc/projects/caesar/aead/morus1280128v1/Morus1280128v1.h \
# EACirc/projects/caesar/aead/morus1280256v1/morus1280256v1_api.h \
# EACirc/projects/caesar/aead/morus1280256v1/morus1280256v1_encrypt.h \
# EACirc/projects/caesar/aead/morus1280256v1/Morus1280256v1.h \
# EACirc/projects/caesar/aead/morus640128v1/morus640128v1_api.h \
# EACirc/projects/caesar/aead/morus640128v1/morus640128v1_encrypt.h \
# EACirc/projects/caesar/aead/morus640128v1/Morus640128v1.h \
# EACirc/projects/caesar/aead/norx3241v1/norx3241v1_api.h \
# EACirc/projects/caesar/aead/norx3241v1/norx3241v1_norx_config.h \
# EACirc/projects/caesar/aead/norx3241v1/norx3241v1_norx.h \
# EACirc/projects/caesar/aead/norx3241v1/norx3241v1_norx_util.h \
# EACirc/projects/caesar/aead/norx3241v1/norx3241v1_encrypt.h \
# EACirc/projects/caesar/aead/norx3241v1/Norx3241v1.h \
# EACirc/projects/caesar/aead/norx3261v1/norx3261v1_api.h \
# EACirc/projects/caesar/aead/norx3261v1/norx3261v1_norx_config.h \
# EACirc/projects/caesar/aead/norx3261v1/norx3261v1_norx.h \
# EACirc/projects/caesar/aead/norx3261v1/norx3261v1_norx_util.h \
# EACirc/projects/caesar/aead/norx3261v1/norx3261v1_encrypt.h \
# EACirc/projects/caesar/aead/norx3261v1/Norx3261v1.h \
# EACirc/projects/caesar/aead/norx6441v1/norx6441v1_api.h \
# EACirc/projects/caesar/aead/norx6441v1/norx6441v1_norx_config.h \
# EACirc/projects/caesar/aead/norx6441v1/norx6441v1_norx.h \
# EACirc/projects/caesar/aead/norx6441v1/norx6441v1_norx_util.h \
# EACirc/projects/caesar/aead/norx6441v1/norx6441v1_encrypt.h \
# EACirc/projects/caesar/aead/norx6441v1/Norx6441v1.h \
# EACirc/projects/caesar/aead/norx6444v1/norx6444v1_api.h \
# EACirc/projects/caesar/aead/norx6444v1/norx6444v1_norx_config.h \
# EACirc/projects/caesar/aead/norx6444v1/norx6444v1_norx.h \
# EACirc/projects/caesar/aead/norx6444v1/norx6444v1_norx_util.h \
# EACirc/projects/caesar/aead/norx6444v1/norx6444v1_encrypt.h \
# EACirc/projects/caesar/aead/norx6444v1/Norx6444v1.h \
# EACirc/projects/caesar/aead/norx6461v1/norx6461v1_api.h \
# EACirc/projects/caesar/aead/norx6461v1/norx6461v1_norx_config.h \
# EACirc/projects/caesar/aead/norx6461v1/norx6461v1_norx.h \
# EACirc/projects/caesar/aead/norx6461v1/norx6461v1_norx_util.h \
# EACirc/projects/caesar/aead/norx6461v1/norx6461v1_encrypt.h \
# EACirc/projects/caesar/aead/norx6461v1/Norx6461v1.h \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_api.h \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_brg_endian.h \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_displayIntermediateValues.h \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_KeccakDuplex.h \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_KeccakF-1600-interface.h \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_KeccakF-1600-reference.h \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_KeccakF-interface.h \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_KeccakP-1600-12-interface.h \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_KeccakP-1600-12-reference.h \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_KeccakParallelDuplex.h \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_Keyak.h \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_NumberOfParallelInstances.h \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_ParallelKeccakFs.h \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_testParallelPaSM.h \
# EACirc/projects/caesar/aead/oceankeyakv1/oceankeyakv1_encrypt.h \
# EACirc/projects/caesar/aead/oceankeyakv1/Oceankeyakv1.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_api.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_inttypes_mingw.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_inttypes_win.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_md32_common.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_omd_api.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_omdsha256.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_sha256.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_stdint_win.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_encrypt.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau128v1/Omdsha256k128n96tau128v1.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_api.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_inttypes_mingw.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_inttypes_win.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_md32_common.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_omd_api.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_omdsha256.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_sha256.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_stdint_win.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_encrypt.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau64v1/Omdsha256k128n96tau64v1.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_api.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_inttypes_mingw.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_inttypes_win.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_md32_common.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_omd_api.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_omdsha256.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_sha256.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_stdint_win.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_encrypt.h \
# EACirc/projects/caesar/aead/omdsha256k128n96tau96v1/Omdsha256k128n96tau96v1.h \
# EACirc/projects/caesar/aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_api.h \
# EACirc/projects/caesar/aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_inttypes_mingw.h \
# EACirc/projects/caesar/aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_inttypes_win.h \
# EACirc/projects/caesar/aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_md32_common.h \
# EACirc/projects/caesar/aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_omd_api.h \
# EACirc/projects/caesar/aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_omdsha256.h \
# EACirc/projects/caesar/aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_sha256.h \
# EACirc/projects/caesar/aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_stdint_win.h \
# EACirc/projects/caesar/aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_encrypt.h \
# EACirc/projects/caesar/aead/omdsha256k192n104tau128v1/Omdsha256k192n104tau128v1.h \
# EACirc/projects/caesar/aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_api.h \
# EACirc/projects/caesar/aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_inttypes_mingw.h \
# EACirc/projects/caesar/aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_inttypes_win.h \
# EACirc/projects/caesar/aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_md32_common.h \
# EACirc/projects/caesar/aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_omd_api.h \
# EACirc/projects/caesar/aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_omdsha256.h \
# EACirc/projects/caesar/aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_sha256.h \
# EACirc/projects/caesar/aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_stdint_win.h \
# EACirc/projects/caesar/aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_encrypt.h \
# EACirc/projects/caesar/aead/omdsha256k256n104tau160v1/Omdsha256k256n104tau160v1.h \
# EACirc/projects/caesar/aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_api.h \
# EACirc/projects/caesar/aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_inttypes_mingw.h \
# EACirc/projects/caesar/aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_inttypes_win.h \
# EACirc/projects/caesar/aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_md32_common.h \
# EACirc/projects/caesar/aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_omd_api.h \
# EACirc/projects/caesar/aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_omdsha256.h \
# EACirc/projects/caesar/aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_sha256.h \
# EACirc/projects/caesar/aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_stdint_win.h \
# EACirc/projects/caesar/aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_encrypt.h \
# EACirc/projects/caesar/aead/omdsha256k256n248tau256v1/Omdsha256k256n248tau256v1.h \
# EACirc/projects/caesar/aead/omdsha512k128n128tau128v1/omdsha512k128n128tau128v1_api.h \
# EACirc/projects/caesar/aead/omdsha512k128n128tau128v1/omdsha512k128n128tau128v1_inttypes_mingw.h \
# EACirc/projects/caesar/aead/omdsha512k128n128tau128v1/omdsha512k128n128tau128v1_inttypes_win.h \
# EACirc/projects/caesar/aead/omdsha512k128n128tau128v1/omdsha512k128n128tau128v1_omd_api.h \
# EACirc/projects/caesar/aead/omdsha512k128n128tau128v1/omdsha512k128n128tau128v1_omdsha512.h \
# EACirc/projects/caesar/aead/omdsha512k128n128tau128v1/omdsha512k128n128tau128v1_sha512.h \
# EACirc/projects/caesar/aead/omdsha512k128n128tau128v1/omdsha512k128n128tau128v1_stdint_win.h \
# EACirc/projects/caesar/aead/omdsha512k128n128tau128v1/omdsha512k128n128tau128v1_encrypt.h \
# EACirc/projects/caesar/aead/omdsha512k128n128tau128v1/Omdsha512k128n128tau128v1.h \
# EACirc/projects/caesar/aead/omdsha512k256n256tau256v1/omdsha512k256n256tau256v1_api.h \
# EACirc/projects/caesar/aead/omdsha512k256n256tau256v1/omdsha512k256n256tau256v1_inttypes_mingw.h \
# EACirc/projects/caesar/aead/omdsha512k256n256tau256v1/omdsha512k256n256tau256v1_inttypes_win.h \
# EACirc/projects/caesar/aead/omdsha512k256n256tau256v1/omdsha512k256n256tau256v1_omd_api.h \
# EACirc/projects/caesar/aead/omdsha512k256n256tau256v1/omdsha512k256n256tau256v1_omdsha512.h \
# EACirc/projects/caesar/aead/omdsha512k256n256tau256v1/omdsha512k256n256tau256v1_sha512.h \
# EACirc/projects/caesar/aead/omdsha512k256n256tau256v1/omdsha512k256n256tau256v1_stdint_win.h \
# EACirc/projects/caesar/aead/omdsha512k256n256tau256v1/omdsha512k256n256tau256v1_encrypt.h \
# EACirc/projects/caesar/aead/omdsha512k256n256tau256v1/Omdsha512k256n256tau256v1.h \
# EACirc/projects/caesar/aead/omdsha512k512n256tau256v1/omdsha512k512n256tau256v1_api.h \
# EACirc/projects/caesar/aead/omdsha512k512n256tau256v1/omdsha512k512n256tau256v1_inttypes_mingw.h \
# EACirc/projects/caesar/aead/omdsha512k512n256tau256v1/omdsha512k512n256tau256v1_inttypes_win.h \
# EACirc/projects/caesar/aead/omdsha512k512n256tau256v1/omdsha512k512n256tau256v1_omd_api.h \
# EACirc/projects/caesar/aead/omdsha512k512n256tau256v1/omdsha512k512n256tau256v1_omdsha512.h \
# EACirc/projects/caesar/aead/omdsha512k512n256tau256v1/omdsha512k512n256tau256v1_sha512.h \
# EACirc/projects/caesar/aead/omdsha512k512n256tau256v1/omdsha512k512n256tau256v1_stdint_win.h \
# EACirc/projects/caesar/aead/omdsha512k512n256tau256v1/omdsha512k512n256tau256v1_encrypt.h \
# EACirc/projects/caesar/aead/omdsha512k512n256tau256v1/Omdsha512k512n256tau256v1.h \
# EACirc/projects/caesar/aead/paeq128/paeq128_api.h \
# EACirc/projects/caesar/aead/paeq128/paeq128_encrypt.h \
# EACirc/projects/caesar/aead/paeq128/Paeq128.h \
# EACirc/projects/caesar/aead/paeq128t/paeq128t_api.h \
# EACirc/projects/caesar/aead/paeq128t/paeq128t_encrypt.h \
# EACirc/projects/caesar/aead/paeq128t/Paeq128t.h \
# EACirc/projects/caesar/aead/paeq128tnm/paeq128tnm_api.h \
# EACirc/projects/caesar/aead/paeq128tnm/paeq128tnm_encrypt.h \
# EACirc/projects/caesar/aead/paeq128tnm/Paeq128tnm.h \
# EACirc/projects/caesar/aead/paeq160/paeq160_api.h \
# EACirc/projects/caesar/aead/paeq160/paeq160_encrypt.h \
# EACirc/projects/caesar/aead/paeq160/Paeq160.h \
# EACirc/projects/caesar/aead/paeq64/paeq64_api.h \
# EACirc/projects/caesar/aead/paeq64/paeq64_encrypt.h \
# EACirc/projects/caesar/aead/paeq64/Paeq64.h \
# EACirc/projects/caesar/aead/paeq80/paeq80_api.h \
# EACirc/projects/caesar/aead/paeq80/paeq80_encrypt.h \
# EACirc/projects/caesar/aead/paeq80/Paeq80.h \
# EACirc/projects/caesar/aead/pi16cipher096v1/pi16cipher096v1_api.h \
# EACirc/projects/caesar/aead/pi16cipher096v1/pi16cipher096v1_encrypt.h \
# EACirc/projects/caesar/aead/pi16cipher096v1/Pi16cipher096v1.h \
# EACirc/projects/caesar/aead/pi16cipher128v1/pi16cipher128v1_api.h \
# EACirc/projects/caesar/aead/pi16cipher128v1/pi16cipher128v1_encrypt.h \
# EACirc/projects/caesar/aead/pi16cipher128v1/Pi16cipher128v1.h \
# EACirc/projects/caesar/aead/pi32cipher128v1/pi32cipher128v1_api.h \
# EACirc/projects/caesar/aead/pi32cipher128v1/pi32cipher128v1_encrypt.h \
# EACirc/projects/caesar/aead/pi32cipher128v1/Pi32cipher128v1.h \
# EACirc/projects/caesar/aead/pi32cipher256v1/pi32cipher256v1_api.h \
# EACirc/projects/caesar/aead/pi32cipher256v1/pi32cipher256v1_encrypt.h \
# EACirc/projects/caesar/aead/pi32cipher256v1/Pi32cipher256v1.h \
# EACirc/projects/caesar/aead/pi64cipher128v1/pi64cipher128v1_api.h \
# EACirc/projects/caesar/aead/pi64cipher128v1/pi64cipher128v1_encrypt.h \
# EACirc/projects/caesar/aead/pi64cipher128v1/Pi64cipher128v1.h \
# EACirc/projects/caesar/aead/pi64cipher256v1/pi64cipher256v1_api.h \
# EACirc/projects/caesar/aead/pi64cipher256v1/pi64cipher256v1_encrypt.h \
# EACirc/projects/caesar/aead/pi64cipher256v1/Pi64cipher256v1.h \
# EACirc/projects/caesar/aead/pi64cipher256v1oneround/pi64cipher256v1oneround_api.h \
# EACirc/projects/caesar/aead/pi64cipher256v1oneround/pi64cipher256v1oneround_encrypt.h \
# EACirc/projects/caesar/aead/pi64cipher256v1oneround/Pi64cipher256v1oneround.h \
# EACirc/projects/caesar/aead/pi64cipher256v1tworounds/pi64cipher256v1tworounds_api.h \
# EACirc/projects/caesar/aead/pi64cipher256v1tworounds/pi64cipher256v1tworounds_encrypt.h \
# EACirc/projects/caesar/aead/pi64cipher256v1tworounds/Pi64cipher256v1tworounds.h \
# EACirc/projects/caesar/aead/polawisv1/polawisv1_encrypt.h \
# EACirc/projects/caesar/aead/polawisv1/Polawisv1.h \
# EACirc/projects/caesar/aead/ppaev11/ppaev11_api.h \
# EACirc/projects/caesar/aead/ppaev11/ppaev11_encrypt.h \
# EACirc/projects/caesar/aead/ppaev11/Ppaev11.h \
# EACirc/projects/caesar/aead/present80n6silcv1/present80n6silcv1_api.h \
# EACirc/projects/caesar/aead/present80n6silcv1/present80n6silcv1_present.h \
# EACirc/projects/caesar/aead/present80n6silcv1/present80n6silcv1_silc.h \
# EACirc/projects/caesar/aead/present80n6silcv1/present80n6silcv1_encrypt.h \
# EACirc/projects/caesar/aead/present80n6silcv1/Present80n6silcv1.h \
# EACirc/projects/caesar/aead/primatesv1ape120/primatesv1ape120_api.h \
# EACirc/projects/caesar/aead/primatesv1ape120/primatesv1ape120_parameters.h \
# EACirc/projects/caesar/aead/primatesv1ape120/primatesv1ape120_primate.h \
# EACirc/projects/caesar/aead/primatesv1ape120/primatesv1ape120_encrypt.h \
# EACirc/projects/caesar/aead/primatesv1ape120/Primatesv1ape120.h \
# EACirc/projects/caesar/aead/primatesv1ape80/primatesv1ape80_api.h \
# EACirc/projects/caesar/aead/primatesv1ape80/primatesv1ape80_parameters.h \
# EACirc/projects/caesar/aead/primatesv1ape80/primatesv1ape80_primate.h \
# EACirc/projects/caesar/aead/primatesv1ape80/primatesv1ape80_encrypt.h \
# EACirc/projects/caesar/aead/primatesv1ape80/Primatesv1ape80.h \
# EACirc/projects/caesar/aead/primatesv1gibbon120/primatesv1gibbon120_api.h \
# EACirc/projects/caesar/aead/primatesv1gibbon120/primatesv1gibbon120_parameters.h \
# EACirc/projects/caesar/aead/primatesv1gibbon120/primatesv1gibbon120_primate.h \
# EACirc/projects/caesar/aead/primatesv1gibbon120/primatesv1gibbon120_encrypt.h \
# EACirc/projects/caesar/aead/primatesv1gibbon120/Primatesv1gibbon120.h \
# EACirc/projects/caesar/aead/primatesv1gibbon80/primatesv1gibbon80_api.h \
# EACirc/projects/caesar/aead/primatesv1gibbon80/primatesv1gibbon80_parameters.h \
# EACirc/projects/caesar/aead/primatesv1gibbon80/primatesv1gibbon80_primate.h \
# EACirc/projects/caesar/aead/primatesv1gibbon80/primatesv1gibbon80_encrypt.h \
# EACirc/projects/caesar/aead/primatesv1gibbon80/Primatesv1gibbon80.h \
# EACirc/projects/caesar/aead/primatesv1hanuman120/primatesv1hanuman120_api.h \
# EACirc/projects/caesar/aead/primatesv1hanuman120/primatesv1hanuman120_parameters.h \
# EACirc/projects/caesar/aead/primatesv1hanuman120/primatesv1hanuman120_primate.h \
# EACirc/projects/caesar/aead/primatesv1hanuman120/primatesv1hanuman120_encrypt.h \
# EACirc/projects/caesar/aead/primatesv1hanuman120/Primatesv1hanuman120.h \
# EACirc/projects/caesar/aead/primatesv1hanuman80/primatesv1hanuman80_api.h \
# EACirc/projects/caesar/aead/primatesv1hanuman80/primatesv1hanuman80_parameters.h \
# EACirc/projects/caesar/aead/primatesv1hanuman80/primatesv1hanuman80_primate.h \
# EACirc/projects/caesar/aead/primatesv1hanuman80/primatesv1hanuman80_encrypt.h \
# EACirc/projects/caesar/aead/primatesv1hanuman80/Primatesv1hanuman80.h \
# EACirc/projects/caesar/aead/proest128apev1/proest128apev1_ape.h \
# EACirc/projects/caesar/aead/proest128apev1/proest128apev1_api.h \
# EACirc/projects/caesar/aead/proest128apev1/proest128apev1_proest128.h \
# EACirc/projects/caesar/aead/proest128apev1/proest128apev1_zerobytes.h \
# EACirc/projects/caesar/aead/proest128apev1/proest128apev1_encrypt.h \
# EACirc/projects/caesar/aead/proest128apev1/Proest128apev1.h \
# EACirc/projects/caesar/aead/proest128copav1/proest128copav1_api.h \
# EACirc/projects/caesar/aead/proest128copav1/proest128copav1_copa.h \
# EACirc/projects/caesar/aead/proest128copav1/proest128copav1_proest128.h \
# EACirc/projects/caesar/aead/proest128copav1/proest128copav1_zerobytes.h \
# EACirc/projects/caesar/aead/proest128copav1/proest128copav1_encrypt.h \
# EACirc/projects/caesar/aead/proest128copav1/Proest128copav1.h \
# EACirc/projects/caesar/aead/proest128otrv1/proest128otrv1_api.h \
# EACirc/projects/caesar/aead/proest128otrv1/proest128otrv1_otr.h \
# EACirc/projects/caesar/aead/proest128otrv1/proest128otrv1_proest128.h \
# EACirc/projects/caesar/aead/proest128otrv1/proest128otrv1_zerobytes.h \
# EACirc/projects/caesar/aead/proest128otrv1/proest128otrv1_encrypt.h \
# EACirc/projects/caesar/aead/proest128otrv1/Proest128otrv1.h \
# EACirc/projects/caesar/aead/proest256apev1/proest256apev1_ape.h \
# EACirc/projects/caesar/aead/proest256apev1/proest256apev1_api.h \
# EACirc/projects/caesar/aead/proest256apev1/proest256apev1_proest256.h \
# EACirc/projects/caesar/aead/proest256apev1/proest256apev1_zerobytes.h \
# EACirc/projects/caesar/aead/proest256apev1/proest256apev1_encrypt.h \
# EACirc/projects/caesar/aead/proest256apev1/Proest256apev1.h \
# EACirc/projects/caesar/aead/proest256copav1/proest256copav1_api.h \
# EACirc/projects/caesar/aead/proest256copav1/proest256copav1_copa.h \
# EACirc/projects/caesar/aead/proest256copav1/proest256copav1_proest256.h \
# EACirc/projects/caesar/aead/proest256copav1/proest256copav1_zerobytes.h \
# EACirc/projects/caesar/aead/proest256copav1/proest256copav1_encrypt.h \
# EACirc/projects/caesar/aead/proest256copav1/Proest256copav1.h \
# EACirc/projects/caesar/aead/proest256otrv1/proest256otrv1_api.h \
# EACirc/projects/caesar/aead/proest256otrv1/proest256otrv1_otr.h \
# EACirc/projects/caesar/aead/proest256otrv1/proest256otrv1_proest256.h \
# EACirc/projects/caesar/aead/proest256otrv1/proest256otrv1_zerobytes.h \
# EACirc/projects/caesar/aead/proest256otrv1/proest256otrv1_encrypt.h \
# EACirc/projects/caesar/aead/proest256otrv1/Proest256otrv1.h \
# EACirc/projects/caesar/aead/raviyoylav1/raviyoylav1_api.h \
# EACirc/projects/caesar/aead/raviyoylav1/raviyoylav1_encrypt.h \
# EACirc/projects/caesar/aead/raviyoylav1/Raviyoylav1.h \
# EACirc/projects/caesar/aead/riverkeyakv1/riverkeyakv1_api.h \
# EACirc/projects/caesar/aead/riverkeyakv1/riverkeyakv1_brg_endian.h \
# EACirc/projects/caesar/aead/riverkeyakv1/riverkeyakv1_displayIntermediateValues.h \
# EACirc/projects/caesar/aead/riverkeyakv1/riverkeyakv1_KeccakDuplex.h \
# EACirc/projects/caesar/aead/riverkeyakv1/riverkeyakv1_KeccakF-800-interface.h \
# EACirc/projects/caesar/aead/riverkeyakv1/riverkeyakv1_KeccakF-800-reference.h \
# EACirc/projects/caesar/aead/riverkeyakv1/riverkeyakv1_KeccakF-interface.h \
# EACirc/projects/caesar/aead/riverkeyakv1/riverkeyakv1_KeccakP-800-12-interface.h \
# EACirc/projects/caesar/aead/riverkeyakv1/riverkeyakv1_KeccakP-800-12-reference.h \
# EACirc/projects/caesar/aead/riverkeyakv1/riverkeyakv1_Keyak.h \
# EACirc/projects/caesar/aead/riverkeyakv1/riverkeyakv1_NumberOfParallelInstances.h \
# EACirc/projects/caesar/aead/riverkeyakv1/riverkeyakv1_encrypt.h \
# EACirc/projects/caesar/aead/riverkeyakv1/Riverkeyakv1.h \
# EACirc/projects/caesar/aead/sablierv1/sablierv1_api.h \
# EACirc/projects/caesar/aead/sablierv1/sablierv1_authenticate_1.h \
# EACirc/projects/caesar/aead/sablierv1/sablierv1_cipher_1.h \
# EACirc/projects/caesar/aead/sablierv1/sablierv1_encrypt.h \
# EACirc/projects/caesar/aead/sablierv1/Sablierv1.h \
# EACirc/projects/caesar/aead/scream10v1/scream10v1_api.h \
# EACirc/projects/caesar/aead/scream10v1/scream10v1_lbox.h \
# EACirc/projects/caesar/aead/scream10v1/scream10v1_params.h \
# EACirc/projects/caesar/aead/scream10v1/scream10v1_sbox.h \
# EACirc/projects/caesar/aead/scream10v1/scream10v1_encrypt.h \
# EACirc/projects/caesar/aead/scream10v1/Scream10v1.h \
# EACirc/projects/caesar/aead/scream10v2/scream10v2_api.h \
# EACirc/projects/caesar/aead/scream10v2/scream10v2_lbox.h \
# EACirc/projects/caesar/aead/scream10v2/scream10v2_params.h \
# EACirc/projects/caesar/aead/scream10v2/scream10v2_sbox.h \
# EACirc/projects/caesar/aead/scream10v2/scream10v2_encrypt.h \
# EACirc/projects/caesar/aead/scream10v2/Scream10v2.h \
# EACirc/projects/caesar/aead/scream12v1/scream12v1_api.h \
# EACirc/projects/caesar/aead/scream12v1/scream12v1_lbox.h \
# EACirc/projects/caesar/aead/scream12v1/scream12v1_params.h \
# EACirc/projects/caesar/aead/scream12v1/scream12v1_sbox.h \
# EACirc/projects/caesar/aead/scream12v1/scream12v1_encrypt.h \
# EACirc/projects/caesar/aead/scream12v1/Scream12v1.h \
# EACirc/projects/caesar/aead/scream12v2/scream12v2_api.h \
# EACirc/projects/caesar/aead/scream12v2/scream12v2_lbox.h \
# EACirc/projects/caesar/aead/scream12v2/scream12v2_params.h \
# EACirc/projects/caesar/aead/scream12v2/scream12v2_sbox.h \
# EACirc/projects/caesar/aead/scream12v2/scream12v2_encrypt.h \
# EACirc/projects/caesar/aead/scream12v2/Scream12v2.h \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_api.h \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_brg_endian.h \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_displayIntermediateValues.h \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_KeccakDuplex.h \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_KeccakF-1600-interface.h \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_KeccakF-1600-reference.h \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_KeccakF-interface.h \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_KeccakP-1600-12-interface.h \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_KeccakP-1600-12-reference.h \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_KeccakParallelDuplex.h \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_Keyak.h \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_NumberOfParallelInstances.h \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_ParallelKeccakFs.h \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_testParallelPaSM.h \
# EACirc/projects/caesar/aead/seakeyakv1/seakeyakv1_encrypt.h \
# EACirc/projects/caesar/aead/seakeyakv1/Seakeyakv1.h \
# EACirc/projects/caesar/aead/shellaes128v1d4n64/shellaes128v1d4n64_aes.h \
# EACirc/projects/caesar/aead/shellaes128v1d4n64/shellaes128v1d4n64_api.h \
# EACirc/projects/caesar/aead/shellaes128v1d4n64/shellaes128v1d4n64_shellaes.h \
# EACirc/projects/caesar/aead/shellaes128v1d4n64/shellaes128v1d4n64_encrypt.h \
# EACirc/projects/caesar/aead/shellaes128v1d4n64/Shellaes128v1d4n64.h \
# EACirc/projects/caesar/aead/shellaes128v1d4n80/shellaes128v1d4n80_aes.h \
# EACirc/projects/caesar/aead/shellaes128v1d4n80/shellaes128v1d4n80_api.h \
# EACirc/projects/caesar/aead/shellaes128v1d4n80/shellaes128v1d4n80_shellaes.h \
# EACirc/projects/caesar/aead/shellaes128v1d4n80/shellaes128v1d4n80_encrypt.h \
# EACirc/projects/caesar/aead/shellaes128v1d4n80/Shellaes128v1d4n80.h \
# EACirc/projects/caesar/aead/shellaes128v1d5n64/shellaes128v1d5n64_aes.h \
# EACirc/projects/caesar/aead/shellaes128v1d5n64/shellaes128v1d5n64_api.h \
# EACirc/projects/caesar/aead/shellaes128v1d5n64/shellaes128v1d5n64_shellaes.h \
# EACirc/projects/caesar/aead/shellaes128v1d5n64/shellaes128v1d5n64_encrypt.h \
# EACirc/projects/caesar/aead/shellaes128v1d5n64/Shellaes128v1d5n64.h \
# EACirc/projects/caesar/aead/shellaes128v1d5n80/shellaes128v1d5n80_aes.h \
# EACirc/projects/caesar/aead/shellaes128v1d5n80/shellaes128v1d5n80_api.h \
# EACirc/projects/caesar/aead/shellaes128v1d5n80/shellaes128v1d5n80_shellaes.h \
# EACirc/projects/caesar/aead/shellaes128v1d5n80/shellaes128v1d5n80_encrypt.h \
# EACirc/projects/caesar/aead/shellaes128v1d5n80/Shellaes128v1d5n80.h \
# EACirc/projects/caesar/aead/shellaes128v1d6n64/shellaes128v1d6n64_aes.h \
# EACirc/projects/caesar/aead/shellaes128v1d6n64/shellaes128v1d6n64_api.h \
# EACirc/projects/caesar/aead/shellaes128v1d6n64/shellaes128v1d6n64_shellaes.h \
# EACirc/projects/caesar/aead/shellaes128v1d6n64/shellaes128v1d6n64_encrypt.h \
# EACirc/projects/caesar/aead/shellaes128v1d6n64/Shellaes128v1d6n64.h \
# EACirc/projects/caesar/aead/shellaes128v1d6n80/shellaes128v1d6n80_aes.h \
# EACirc/projects/caesar/aead/shellaes128v1d6n80/shellaes128v1d6n80_api.h \
# EACirc/projects/caesar/aead/shellaes128v1d6n80/shellaes128v1d6n80_shellaes.h \
# EACirc/projects/caesar/aead/shellaes128v1d6n80/shellaes128v1d6n80_encrypt.h \
# EACirc/projects/caesar/aead/shellaes128v1d6n80/Shellaes128v1d6n80.h \
# EACirc/projects/caesar/aead/shellaes128v1d7n64/shellaes128v1d7n64_aes.h \
# EACirc/projects/caesar/aead/shellaes128v1d7n64/shellaes128v1d7n64_api.h \
# EACirc/projects/caesar/aead/shellaes128v1d7n64/shellaes128v1d7n64_shellaes.h \
# EACirc/projects/caesar/aead/shellaes128v1d7n64/shellaes128v1d7n64_encrypt.h \
# EACirc/projects/caesar/aead/shellaes128v1d7n64/Shellaes128v1d7n64.h \
# EACirc/projects/caesar/aead/shellaes128v1d7n80/shellaes128v1d7n80_aes.h \
# EACirc/projects/caesar/aead/shellaes128v1d7n80/shellaes128v1d7n80_api.h \
# EACirc/projects/caesar/aead/shellaes128v1d7n80/shellaes128v1d7n80_shellaes.h \
# EACirc/projects/caesar/aead/shellaes128v1d7n80/shellaes128v1d7n80_encrypt.h \
# EACirc/projects/caesar/aead/shellaes128v1d7n80/Shellaes128v1d7n80.h \
# EACirc/projects/caesar/aead/shellaes128v1d8n64/shellaes128v1d8n64_aes.h \
# EACirc/projects/caesar/aead/shellaes128v1d8n64/shellaes128v1d8n64_api.h \
# EACirc/projects/caesar/aead/shellaes128v1d8n64/shellaes128v1d8n64_shellaes.h \
# EACirc/projects/caesar/aead/shellaes128v1d8n64/shellaes128v1d8n64_encrypt.h \
# EACirc/projects/caesar/aead/shellaes128v1d8n64/Shellaes128v1d8n64.h \
# EACirc/projects/caesar/aead/shellaes128v1d8n80/shellaes128v1d8n80_aes.h \
# EACirc/projects/caesar/aead/shellaes128v1d8n80/shellaes128v1d8n80_api.h \
# EACirc/projects/caesar/aead/shellaes128v1d8n80/shellaes128v1d8n80_shellaes.h \
# EACirc/projects/caesar/aead/shellaes128v1d8n80/shellaes128v1d8n80_encrypt.h \
# EACirc/projects/caesar/aead/shellaes128v1d8n80/Shellaes128v1d8n80.h \
# EACirc/projects/caesar/aead/silverv1/silverv1_api.h \
# EACirc/projects/caesar/aead/silverv1/silverv1_rijndaelEndianNeutral.h \
# EACirc/projects/caesar/aead/silverv1/silverv1_encrypt.h \
# EACirc/projects/caesar/aead/silverv1/Silverv1.h \
# EACirc/projects/caesar/aead/stribob192r1/stribob192r1_api.h \
# EACirc/projects/caesar/aead/stribob192r1/stribob192r1_crypto_aead.h \
# EACirc/projects/caesar/aead/stribob192r1/stribob192r1_stribob.h \
# EACirc/projects/caesar/aead/stribob192r1/stribob192r1_encrypt.h \
# EACirc/projects/caesar/aead/stribob192r1/Stribob192r1.h \
# EACirc/projects/caesar/aead/tiaoxinv1/tiaoxinv1_aes_round.h \
# EACirc/projects/caesar/aead/tiaoxinv1/tiaoxinv1_api.h \
# EACirc/projects/caesar/aead/tiaoxinv1/tiaoxinv1_tiaoxin-reference.h \
# EACirc/projects/caesar/aead/tiaoxinv1/tiaoxinv1_encrypt.h \
# EACirc/projects/caesar/aead/tiaoxinv1/Tiaoxinv1.h \
# EACirc/projects/caesar/aead/trivia0v1/trivia0v1_api.h \
# EACirc/projects/caesar/aead/trivia0v1/trivia0v1_encrypt.h \
# EACirc/projects/caesar/aead/trivia0v1/Trivia0v1.h \
# EACirc/projects/caesar/aead/trivia128v1/trivia128v1_api.h \
# EACirc/projects/caesar/aead/trivia128v1/trivia128v1_encrypt.h \
# EACirc/projects/caesar/aead/trivia128v1/Trivia128v1.h \
# EACirc/projects/caesar/aead/twine80n6clocv1/twine80n6clocv1_api.h \
# EACirc/projects/caesar/aead/twine80n6clocv1/twine80n6clocv1_cloc.h \
# EACirc/projects/caesar/aead/twine80n6clocv1/twine80n6clocv1_twine.h \
# EACirc/projects/caesar/aead/twine80n6clocv1/twine80n6clocv1_encrypt.h \
# EACirc/projects/caesar/aead/twine80n6clocv1/Twine80n6clocv1.h \
# EACirc/projects/caesar/aead/wheeshtv1mr3fr1t128/wheeshtv1mr3fr1t128_api.h \
# EACirc/projects/caesar/aead/wheeshtv1mr3fr1t128/wheeshtv1mr3fr1t128_encrypt.h \
# EACirc/projects/caesar/aead/wheeshtv1mr3fr1t128/wheeshtv1mr3fr1t128_encrypt.h \
# EACirc/projects/caesar/aead/wheeshtv1mr3fr1t128/Wheeshtv1mr3fr1t128.h \
# EACirc/projects/caesar/aead/wheeshtv1mr3fr1t256/wheeshtv1mr3fr1t256_api.h \
# EACirc/projects/caesar/aead/wheeshtv1mr3fr1t256/wheeshtv1mr3fr1t256_encrypt.h \
# EACirc/projects/caesar/aead/wheeshtv1mr3fr1t256/wheeshtv1mr3fr1t256_encrypt.h \
# EACirc/projects/caesar/aead/wheeshtv1mr3fr1t256/Wheeshtv1mr3fr1t256.h \
# EACirc/projects/caesar/aead/wheeshtv1mr3fr3t256/wheeshtv1mr3fr3t256_api.h \
# EACirc/projects/caesar/aead/wheeshtv1mr3fr3t256/wheeshtv1mr3fr3t256_encrypt.h \
# EACirc/projects/caesar/aead/wheeshtv1mr3fr3t256/wheeshtv1mr3fr3t256_encrypt.h \
# EACirc/projects/caesar/aead/wheeshtv1mr3fr3t256/Wheeshtv1mr3fr3t256.h \
# EACirc/projects/caesar/aead/wheeshtv1mr5fr7t256/wheeshtv1mr5fr7t256_api.h \
# EACirc/projects/caesar/aead/wheeshtv1mr5fr7t256/wheeshtv1mr5fr7t256_encrypt.h \
# EACirc/projects/caesar/aead/wheeshtv1mr5fr7t256/wheeshtv1mr5fr7t256_encrypt.h \
# EACirc/projects/caesar/aead/wheeshtv1mr5fr7t256/Wheeshtv1mr5fr7t256.h \
# EACirc/projects/caesar/aead/yaes128v2/yaes128v2_aes-128.h \
# EACirc/projects/caesar/aead/yaes128v2/yaes128v2_api.h \
# EACirc/projects/caesar/aead/yaes128v2/yaes128v2_encrypt.h \
# EACirc/projects/caesar/aead/yaes128v2/yaes128v2_encrypt.h \
# EACirc/projects/caesar/aead/yaes128v2/Yaes128v2.h \

OTHER_FILES += \
 EACirc/projects/caesar/aead/acorn128/About.md \
 EACirc/projects/caesar/aead/aeadaes128ocbtaglen128v1/About.md \
 EACirc/projects/caesar/aead/aeadaes128ocbtaglen64v1/About.md \
 EACirc/projects/caesar/aead/aeadaes128ocbtaglen96v1/About.md \
 EACirc/projects/caesar/aead/aeadaes192ocbtaglen128v1/About.md \
 EACirc/projects/caesar/aead/aeadaes192ocbtaglen64v1/About.md \
 EACirc/projects/caesar/aead/aeadaes192ocbtaglen96v1/About.md \
 EACirc/projects/caesar/aead/aeadaes256ocbtaglen128v1/About.md \
 EACirc/projects/caesar/aead/aeadaes256ocbtaglen64v1/About.md \
 EACirc/projects/caesar/aead/aeadaes256ocbtaglen96v1/About.md \
 EACirc/projects/caesar/aead/aegis128/About.md \
 EACirc/projects/caesar/aead/aegis128l/About.md \
 EACirc/projects/caesar/aead/aegis256/About.md \
# EACirc/projects/caesar/aead/aes128avalanchev1/About.md \ # compilaation ok, sigsegv while running
# EACirc/projects/caesar/aead/aes128cpfbv1/About.md \
 EACirc/projects/caesar/aead/aes128gcmv1/About.md \
 EACirc/projects/caesar/aead/aes128marble4rv1/About.md \
 EACirc/projects/caesar/aead/aes128n12clocv1/About.md \
# EACirc/projects/caesar/aead/aes128n12silcv1/About.md \
 EACirc/projects/caesar/aead/aes128n8clocv1/About.md \
# EACirc/projects/caesar/aead/aes128n8silcv1/About.md \
 EACirc/projects/caesar/aead/aes128otrpv1/About.md \
 EACirc/projects/caesar/aead/aes128otrsv1/About.md \
 EACirc/projects/caesar/aead/aes128poetv1aes128/About.md \
 EACirc/projects/caesar/aead/aes128poetv1aes4/About.md \
 EACirc/projects/caesar/aead/aes192avalanchev1/About.md \
 EACirc/projects/caesar/aead/aes256avalanchev1/About.md \
 EACirc/projects/caesar/aead/aes256cpfbv1/About.md \
 EACirc/projects/caesar/aead/aes256gcmv1/About.md \
# EACirc/projects/caesar/aead/aes256otrpv1/About.md \
# EACirc/projects/caesar/aead/aes256otrsv1/About.md \
# EACirc/projects/caesar/aead/aescopav1/About.md \
# EACirc/projects/caesar/aead/aesjambuv1/About.md \
# EACirc/projects/caesar/aead/aezv1/About.md \
# EACirc/projects/caesar/aead/aezv3/About.md \
# EACirc/projects/caesar/aead/artemia128v1/About.md \
# EACirc/projects/caesar/aead/artemia256v1/About.md \
# EACirc/projects/caesar/aead/ascon128v1/About.md \
# EACirc/projects/caesar/aead/ascon96v1/About.md \
# EACirc/projects/caesar/aead/calicov8/About.md \
# EACirc/projects/caesar/aead/cba1/About.md \
# EACirc/projects/caesar/aead/cba10/About.md \
# EACirc/projects/caesar/aead/cba2/About.md \
# EACirc/projects/caesar/aead/cba3/About.md \
# EACirc/projects/caesar/aead/cba4/About.md \
# EACirc/projects/caesar/aead/cba5/About.md \
# EACirc/projects/caesar/aead/cba6/About.md \
# EACirc/projects/caesar/aead/cba7/About.md \
# EACirc/projects/caesar/aead/cba8/About.md \
# EACirc/projects/caesar/aead/cba9/About.md \
# EACirc/projects/caesar/aead/cmcc22v1/About.md \
# EACirc/projects/caesar/aead/cmcc24v1/About.md \
# EACirc/projects/caesar/aead/cmcc42v1/About.md \
# EACirc/projects/caesar/aead/cmcc44v1/About.md \
# EACirc/projects/caesar/aead/cmcc84v1/About.md \
# EACirc/projects/caesar/aead/deoxyseq128128v1/About.md \
# EACirc/projects/caesar/aead/deoxyseq256128v1/About.md \
# EACirc/projects/caesar/aead/deoxysneq128128v1/About.md \
# EACirc/projects/caesar/aead/deoxysneq256128v1/About.md \
# EACirc/projects/caesar/aead/elmd1000v1/About.md \
# EACirc/projects/caesar/aead/elmd1001v1/About.md \
# EACirc/projects/caesar/aead/elmd101270v1/About.md \
# EACirc/projects/caesar/aead/elmd101271v1/About.md \
# EACirc/projects/caesar/aead/elmd500v1/About.md \
# EACirc/projects/caesar/aead/elmd501v1/About.md \
# EACirc/projects/caesar/aead/elmd51270v1/About.md \
# EACirc/projects/caesar/aead/elmd51271v1/About.md \
# EACirc/projects/caesar/aead/enchilada128v1/About.md \
# EACirc/projects/caesar/aead/enchilada256v1/About.md \
# EACirc/projects/caesar/aead/hs1sivhiv1/About.md \
# EACirc/projects/caesar/aead/hs1sivlov1/About.md \
# EACirc/projects/caesar/aead/hs1sivv1/About.md \
# EACirc/projects/caesar/aead/icepole128av1/About.md \
# EACirc/projects/caesar/aead/icepole128v1/About.md \
# EACirc/projects/caesar/aead/icepole256av1/About.md \
# EACirc/projects/caesar/aead/ifeedaes128n104v1/About.md \
# EACirc/projects/caesar/aead/ifeedaes128n96v1/About.md \
# EACirc/projects/caesar/aead/iscream12v1/About.md \
# EACirc/projects/caesar/aead/iscream12v2/About.md \
# EACirc/projects/caesar/aead/iscream14v1/About.md \
# EACirc/projects/caesar/aead/iscream14v2/About.md \
# EACirc/projects/caesar/aead/joltikeq12864v1/About.md \
# EACirc/projects/caesar/aead/joltikeq6464v1/About.md \
# EACirc/projects/caesar/aead/joltikeq8048v1/About.md \
# EACirc/projects/caesar/aead/joltikeq9696v1/About.md \
# EACirc/projects/caesar/aead/joltikneq12864v1/About.md \
# EACirc/projects/caesar/aead/joltikneq6464v1/About.md \
# EACirc/projects/caesar/aead/joltikneq8048v1/About.md \
# EACirc/projects/caesar/aead/joltikneq9696v1/About.md \
# EACirc/projects/caesar/aead/juliusv1draft/About.md \
# EACirc/projects/caesar/aead/ketjejrv1/About.md \
# EACirc/projects/caesar/aead/ketjesrv1/About.md \
# EACirc/projects/caesar/aead/kiasueq128v1/About.md \
# EACirc/projects/caesar/aead/kiasuneq128v1/About.md \
# EACirc/projects/caesar/aead/lacv1/About.md \
# EACirc/projects/caesar/aead/lakekeyakv1/About.md \
# EACirc/projects/caesar/aead/led80n6silcv1/About.md \
# EACirc/projects/caesar/aead/minalpherv1/About.md \
# EACirc/projects/caesar/aead/morus1280128v1/About.md \
# EACirc/projects/caesar/aead/morus1280256v1/About.md \
# EACirc/projects/caesar/aead/morus640128v1/About.md \
# EACirc/projects/caesar/aead/norx3241v1/About.md \
# EACirc/projects/caesar/aead/norx3261v1/About.md \
# EACirc/projects/caesar/aead/norx6441v1/About.md \
# EACirc/projects/caesar/aead/norx6444v1/About.md \
# EACirc/projects/caesar/aead/norx6461v1/About.md \
# EACirc/projects/caesar/aead/oceankeyakv1/About.md \
# EACirc/projects/caesar/aead/omdsha256k128n96tau128v1/About.md \
# EACirc/projects/caesar/aead/omdsha256k128n96tau64v1/About.md \
# EACirc/projects/caesar/aead/omdsha256k128n96tau96v1/About.md \
# EACirc/projects/caesar/aead/omdsha256k192n104tau128v1/About.md \
# EACirc/projects/caesar/aead/omdsha256k256n104tau160v1/About.md \
# EACirc/projects/caesar/aead/omdsha256k256n248tau256v1/About.md \
# EACirc/projects/caesar/aead/omdsha512k128n128tau128v1/About.md \
# EACirc/projects/caesar/aead/omdsha512k256n256tau256v1/About.md \
# EACirc/projects/caesar/aead/omdsha512k512n256tau256v1/About.md \
# EACirc/projects/caesar/aead/paeq128/About.md \
# EACirc/projects/caesar/aead/paeq128t/About.md \
# EACirc/projects/caesar/aead/paeq128tnm/About.md \
# EACirc/projects/caesar/aead/paeq160/About.md \
# EACirc/projects/caesar/aead/paeq64/About.md \
# EACirc/projects/caesar/aead/paeq80/About.md \
# EACirc/projects/caesar/aead/pi16cipher096v1/About.md \
# EACirc/projects/caesar/aead/pi16cipher128v1/About.md \
# EACirc/projects/caesar/aead/pi32cipher128v1/About.md \
# EACirc/projects/caesar/aead/pi32cipher256v1/About.md \
# EACirc/projects/caesar/aead/pi64cipher128v1/About.md \
# EACirc/projects/caesar/aead/pi64cipher256v1/About.md \
# EACirc/projects/caesar/aead/pi64cipher256v1oneround/About.md \
# EACirc/projects/caesar/aead/pi64cipher256v1tworounds/About.md \
# EACirc/projects/caesar/aead/polawisv1/About.md \
# EACirc/projects/caesar/aead/ppaev11/About.md \
# EACirc/projects/caesar/aead/present80n6silcv1/About.md \
# EACirc/projects/caesar/aead/primatesv1ape120/About.md \
# EACirc/projects/caesar/aead/primatesv1ape80/About.md \
# EACirc/projects/caesar/aead/primatesv1gibbon120/About.md \
# EACirc/projects/caesar/aead/primatesv1gibbon80/About.md \
# EACirc/projects/caesar/aead/primatesv1hanuman120/About.md \
# EACirc/projects/caesar/aead/primatesv1hanuman80/About.md \
# EACirc/projects/caesar/aead/proest128apev1/About.md \
# EACirc/projects/caesar/aead/proest128copav1/About.md \
# EACirc/projects/caesar/aead/proest128otrv1/About.md \
# EACirc/projects/caesar/aead/proest256apev1/About.md \
# EACirc/projects/caesar/aead/proest256copav1/About.md \
# EACirc/projects/caesar/aead/proest256otrv1/About.md \
# EACirc/projects/caesar/aead/raviyoylav1/About.md \
# EACirc/projects/caesar/aead/riverkeyakv1/About.md \
# EACirc/projects/caesar/aead/sablierv1/About.md \
# EACirc/projects/caesar/aead/scream10v1/About.md \
# EACirc/projects/caesar/aead/scream10v2/About.md \
# EACirc/projects/caesar/aead/scream12v1/About.md \
# EACirc/projects/caesar/aead/scream12v2/About.md \
# EACirc/projects/caesar/aead/seakeyakv1/About.md \
# EACirc/projects/caesar/aead/shellaes128v1d4n64/About.md \
# EACirc/projects/caesar/aead/shellaes128v1d4n80/About.md \
# EACirc/projects/caesar/aead/shellaes128v1d5n64/About.md \
# EACirc/projects/caesar/aead/shellaes128v1d5n80/About.md \
# EACirc/projects/caesar/aead/shellaes128v1d6n64/About.md \
# EACirc/projects/caesar/aead/shellaes128v1d6n80/About.md \
# EACirc/projects/caesar/aead/shellaes128v1d7n64/About.md \
# EACirc/projects/caesar/aead/shellaes128v1d7n80/About.md \
# EACirc/projects/caesar/aead/shellaes128v1d8n64/About.md \
# EACirc/projects/caesar/aead/shellaes128v1d8n80/About.md \
# EACirc/projects/caesar/aead/silverv1/About.md \
# EACirc/projects/caesar/aead/stribob192r1/About.md \
# EACirc/projects/caesar/aead/tiaoxinv1/About.md \
# EACirc/projects/caesar/aead/trivia0v1/About.md \
# EACirc/projects/caesar/aead/trivia128v1/About.md \
# EACirc/projects/caesar/aead/twine80n6clocv1/About.md \
# EACirc/projects/caesar/aead/wheeshtv1mr3fr1t128/About.md \
# EACirc/projects/caesar/aead/wheeshtv1mr3fr1t256/About.md \
# EACirc/projects/caesar/aead/wheeshtv1mr3fr3t256/About.md \
# EACirc/projects/caesar/aead/wheeshtv1mr5fr7t256/About.md \
# EACirc/projects/caesar/aead/yaes128v2/About.md \
