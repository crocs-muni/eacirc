TEMPLATE = app
CONFIG += console
CONFIG -= qt

SUPPRESSED_WARNINGS = -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable \
                      -Wno-unused-function -Wno-unused-value
unix{
    QMAKE_CXX = g++-4.7
}
QMAKE_LFLAGS_RELEASE += -static -static-libgcc -static-libstdc++
QMAKE_CXXFLAGS += -std=c++11 $$SUPPRESSED_WARNINGS -Wall -Wextra #-Weffc++
QMAKE_CXXFLAGS += -isystem ../EACirc/galib -isystem ../EACirc/tinyXML
INCLUDEPATH += ./EACirc ./EACirc/galib ./EACirc/tinyXML

# to load GAlib and tinyXML as external libraries, do following:
# - comment the Libraries section in SOURCES and HEADERS
# - change includes in all files from "galib/header.h" to "header.h"
#   (same for tinyXML) - search for //libinclude in source files
# - uncomment following line (adding the libraries)
# LIBS += -ltinyXML -L../EACirc/tinyXML -lga -L../EACirc/ga

SOURCES += \
# === Main project files ===
    EACirc/Main.cpp \
    EACirc/CircuitGenome.cpp \
    EACirc/CommonFnc.cpp \
    EACirc/XMLProcessor.cpp \
    EACirc/Logger.cpp \
    EACirc/status.cpp \
    EACirc/EACirc.cpp \
    EACirc/circuit_evaluator/DistinguishTwoEvaluator.cpp \
    EACirc/circuit_evaluator/ICircuitEvaluator.cpp \
    EACirc/circuit_evaluator/PredictAvalancheEvaluator.cpp \
    EACirc/circuit_evaluator/PredictBitCircuitEvaluator.cpp \
    EACirc/circuit_evaluator/PredictBitGroupParityCircuitEvaluator.cpp \
    EACirc/circuit_evaluator/PredictByteCircuitEvaluator.cpp \
    EACirc/circuit_evaluator/PredictBytesParityCircuitEvaluator.cpp \
    EACirc/circuit_evaluator/PredictHammingWeightCircuitEvaluator.cpp \
    EACirc/random_generator/BiasRndGen.cpp \
    EACirc/random_generator/IRndGen.cpp \
    EACirc/random_generator/QuantumRndGen.cpp \
    EACirc/random_generator/MD5RndGen.cpp \
    EACirc/random_generator/md5.cpp \
    EACirc/projects/estream/EncryptorDecryptor.cpp \
# === testing using CATCH framework ===
    EACirc/self_tests/tests.cpp \
# === eSTREAM cipher files ===
    EACirc/projects/estream/estreamInterface.cpp \
    EACirc/projects/estream/ciphers/zk-crypt/zk-crypt-v3.cpp \   # not used
    EACirc/projects/estream/ciphers/yamb/yamb.cpp \
    EACirc/projects/estream/ciphers/wg/wg.cpp \
    EACirc/projects/estream/ciphers/tsc-4/tsc-4.cpp \
    EACirc/projects/estream/ciphers/trivium/trivium.cpp \
    EACirc/projects/estream/ciphers/sosemanuk/sosemanuk.cpp \
    EACirc/projects/estream/ciphers/sfinks/sfinks.cpp \
    EACirc/projects/estream/ciphers/salsa20/salsa20.cpp \
    EACirc/projects/estream/ciphers/rabbit/rabbit.cpp \
    EACirc/projects/estream/ciphers/py/py6.cpp \
    EACirc/projects/estream/ciphers/pomaranch/pomaranch.cpp \
#    EACirc/projects/estream/ciphers/polarbear/polar-bear.cpp \  # not implemented in EncryptorDecryptor
#    EACirc/projects/estream/ciphers/polarbear/aescrypt.cpp \    # do not include!
#    EACirc/projects/estream/ciphers/polarbear/aestab.cpp \      # do not include!
#    EACirc/projects/estream/ciphers/polarbear/whirltab.cpp \    # do not include!
#    EACirc/projects/estream/ciphers/nls/nlsref.cpp \            # not implemented in EncryptorDecryptor
#    EACirc/projects/estream/ciphers/moustique/moustique.cpp \   # not implemented in EncryptorDecryptor
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
#    EACirc/projects/estream/ciphers/decim/decim-128.c \         # do not include!
    EACirc/projects/estream/ciphers/cryptmt/cryptmt-v3.cpp \     # not used
#    EACirc/projects/estream/ciphers/cryptmt/altivec.cpp \       # do not include!
#    EACirc/projects/estream/ciphers/cryptmt/sse2.cpp            # do not include!
    EACirc/projects/estream/ciphers/achterbahn/achterbahn-128-80.cpp \
    EACirc/projects/estream/ciphers/abc/abc-v3.cpp \             # not used
# === Libraries (redundant if using pre-compiled) ===
    EACirc/tinyXML/tinystr.cpp EACirc/tinyXML/tinyxml.cpp EACirc/tinyXML/tinyxmlerror.cpp EACirc/tinyXML/tinyxmlparser.cpp \
    EACirc/galib/GA1DArrayGenome.cpp EACirc/galib/GA1DBinStrGenome.cpp EACirc/galib/GA2DArrayGenome.cpp EACirc/galib/GA2DBinStrGenome.cpp \
    EACirc/galib/GA3DArrayGenome.cpp EACirc/galib/GA3DBinStrGenome.cpp EACirc/galib/GAAllele.cpp EACirc/galib/GABaseGA.cpp EACirc/galib/GABin2DecGenome.cpp \
    EACirc/galib/gabincvt.cpp EACirc/galib/GABinStr.cpp EACirc/galib/GADCrowdingGA.cpp EACirc/galib/GADemeGA.cpp EACirc/galib/gaerror.cpp EACirc/galib/GAGenome.cpp \
    EACirc/galib/GAIncGA.cpp EACirc/galib/GAList.cpp EACirc/galib/GAListBASE.cpp EACirc/galib/GAListGenome.cpp EACirc/galib/GAParameter.cpp EACirc/galib/GAPopulation.cpp \
    EACirc/galib/garandom.cpp EACirc/galib/GARealGenome.cpp EACirc/galib/GAScaling.cpp EACirc/galib/GASelector.cpp EACirc/galib/GASimpleGA.cpp EACirc/galib/GASStateGA.cpp \
    EACirc/galib/GAStatistics.cpp EACirc/galib/GAStringGenome.cpp EACirc/galib/GATree.cpp EACirc/galib/GATreeBASE.cpp EACirc/galib/GATreeGenome.cpp \
    EACirc/standalone_testers/TestDistinctorCircuit.cpp \
    EACirc/projects/IProject.cpp \
    EACirc/projects/estream/EstreamProject.cpp \
    EACirc/projects/sha3/Sha3Project.cpp

HEADERS += \
# === Main project files ===
    EACirc/Main.h \
    EACirc/CircuitGenome.h \
    EACirc/CommonFnc.h \
    EACirc/XMLProcessor.h \
    EACirc/EACconstants.h \
    EACirc/EACglobals.h \
    EACirc/Logger.h \
    EACirc/status.h \
    EACirc/EACirc.h \
    EACirc/EAC_circuit.h \
    EACirc/circuit_evaluator/DistinguishTwoEvaluator.h \
    EACirc/circuit_evaluator/ICircuitEvaluator.h \
    EACirc/circuit_evaluator/PredictAvalancheEvaluator.h \
    EACirc/circuit_evaluator/PredictBitCircuitEvaluator.h \
    EACirc/circuit_evaluator/PredictBitGroupParityCircuitEvaluator.h \
    EACirc/circuit_evaluator/PredictByteCircuitEvaluator.h \
    EACirc/circuit_evaluator/PredictBytesParityCircuitEvaluator.h \
    EACirc/circuit_evaluator/PredictHammingWeightCircuitEvaluator.h \
    EACirc/standalone_testers/TestDistinctorCircuit.h \
    EACirc/random_generator/BiasRndGen.h \
    EACirc/random_generator/IRndGen.h \
    EACirc/random_generator/QuantumRndGen.h \
    EACirc/random_generator/MD5RndGen.h \
    EACirc/random_generator/md5.h \
    EACirc/projects/estream/EncryptorDecryptor.h \
# === CATCH testing framework ===
    EACirc/self_tests/catch.hpp \
    EACirc/self_tests/tests.h \
# === eSTREAM cipher files ===
    EACirc/projects/estream/estreamInterface.h \
    EACirc/projects/estream/ciphers/ecrypt-config.h \
    EACirc/projects/estream/ciphers/ecrypt-machine.h \
    EACirc/projects/estream/ciphers/ecrypt-portable.h \
    EACirc/projects/estream/ciphers/zk-crypt/ecrypt-sync.h \     # not used
    EACirc/projects/estream/ciphers/zk-crypt/ZKdef.h \           # not used
    EACirc/projects/estream/ciphers/zk-crypt/ZKengine.h \        # not used
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
#    EACirc/projects/estream/ciphers/polarbear/aes.h \           # not implemented in EncryptorDecryptor
#    EACirc/projects/estream/ciphers/polarbear/aesopt.h \        # not implemented in EncryptorDecryptor
#    EACirc/projects/estream/ciphers/polarbear/ecrypt-sync.h \   # not implemented in EncryptorDecryptor
#    EACirc/projects/estream/ciphers/nls/ecrypt-sync.h \         # not implemented in EncryptorDecryptor
#    EACirc/projects/estream/ciphers/nls/ecrypt-sync-ae.h \      # not implemented in EncryptorDecryptor
#    EACirc/projects/estream/ciphers/nls/nls.h \                 # not implemented in EncryptorDecryptor
#    EACirc/projects/estream/ciphers/nls/nlsmultab.h \           # not implemented in EncryptorDecryptor
#    EACirc/projects/estream/ciphers/nls/nlssbox.h \             # not implemented in EncryptorDecryptor
#    EACirc/projects/estream/ciphers/moustique/ecrypt-ssyn.h \   # not implemented in EncryptorDecryptor
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
#    EACirc/projects/estream/ciphers/decim/decim-128.h        # do not include!
    EACirc/projects/estream/ciphers/decim/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/cryptmt/ecrypt-sync.h \   # not used
    EACirc/projects/estream/ciphers/cryptmt/params.h \        # not used
    EACirc/projects/estream/ciphers/achterbahn/achterbahn.h \
    EACirc/projects/estream/ciphers/achterbahn/ecrypt-sync.h \
    EACirc/projects/estream/ciphers/abc/abc.h \               # not used
    EACirc/projects/estream/ciphers/abc/abc-tables.h \        # not used
    EACirc/projects/estream/ciphers/abc/ecrypt-sync.h \       # not used
# === Libraries (redundant if using pre-compiled) ===
    EACirc/tinyXML/tinystr.h EACirc/tinyXML/tinyxml.h \
    EACirc/galib/GA1DArrayGenome.h EACirc/galib/ga.h EACirc/galib/GA1DBinStrGenome.h EACirc/galib/GA2DArrayGenome.h EACirc/galib/GA2DBinStrGenome.h \
    EACirc/galib/GA3DArrayGenome.h EACirc/galib/GA3DBinStrGenome.h EACirc/galib/GAAllele.h EACirc/galib/GAArray.h EACirc/galib/GABaseGA.h \
    EACirc/galib/GABin2DecGenome.h EACirc/galib/gabincvt.h EACirc/galib/GABinStr.h EACirc/galib/gaconfig.h EACirc/galib/GADCrowdingGA.h EACirc/galib/GADemeGA.h \
    EACirc/galib/gaerror.h EACirc/galib/GAEvalData.h EACirc/galib/GAGenome.h EACirc/galib/gaid.h EACirc/galib/GAIncGA.h EACirc/galib/GAList.h EACirc/galib/GAListBASE.h \
    EACirc/galib/GAListGenome.h EACirc/galib/GAMask.h EACirc/galib/GANode.h EACirc/galib/GAParameter.h EACirc/galib/GAPopulation.h EACirc/galib/garandom.h \
    EACirc/galib/GARealGenome.h EACirc/galib/GAScaling.h EACirc/galib/GASelector.h EACirc/galib/GASimpleGA.h EACirc/galib/GASStateGA.h EACirc/galib/GAStatistics.h \
    EACirc/galib/GAStringGenome.h EACirc/galib/GATree.h EACirc/galib/GATreeBASE.h EACirc/galib/GATreeGenome.h EACirc/galib/gatypes.h EACirc/galib/gaversion.h \
    EACirc/galib/std_stream.h \
    EACirc/projects/IProject.h \
    EACirc/projects/estream/EstreamProject.h \
    EACirc/projects/sha3/Sha3Project.h \
    EACirc/projects/estream/EstreamConstants.h

OTHER_FILES += \
    EACirc/config.xml
