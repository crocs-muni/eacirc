TEMPLATE = app
CONFIG += console
CONFIG -= qt

SUPPRESSED_WARNINGS = -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable \
                      -Wno-unused-function -Wno-unused-value
unix{
    QMAKE_CXX = g++-4.7
}
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
    EACirc/CircuitGenome.cpp \
    EACirc/CommonFnc.cpp \
    EACirc/XMLProcessor.cpp \
    EACirc/Logger.cpp \
    EACirc/Evaluator.cpp \
    EACirc/status.cpp \
    EACirc/EACirc.cpp \
    EACirc/test_vector_generator/ITestVectGener.cpp \
    EACirc/test_vector_generator/EstreamTestVectGener.cpp \
    EACirc/circuit_evaluator/DistinguishTwoEvaluator.cpp \
    EACirc/circuit_evaluator/ICircuitEvaluator.cpp \
    EACirc/circuit_evaluator/PredictAvalancheEvaluator.cpp \
    EACirc/circuit_evaluator/PredictBitCircuitEvaluator.cpp \
    EACirc/circuit_evaluator/PredictBitGroupParityCircuitEvaluator.cpp \
    EACirc/circuit_evaluator/PredictByteCircuitEvaluator.cpp \
    EACirc/circuit_evaluator/PredictBytesParityCircuitEvaluator.cpp \
    EACirc/circuit_evaluator/PredictHammingWeightCircuitEvaluator.cpp \
    EACirc/standalone_testers/TestDisctinctorCircuit.cpp \
    EACirc/random_generator/BiasRndGen.cpp \
    EACirc/random_generator/IRndGen.cpp \
    EACirc/random_generator/QuantumRndGen.cpp \
    EACirc/estream/EncryptorDecryptor.cpp \
# === eSTREAM cipher files ===
    EACirc/estream/estreamInterface.cpp \
    EACirc/estream/ciphers/zk-crypt/zk-crypt-v3.cpp \   # not used
    EACirc/estream/ciphers/yamb/yamb.cpp \
    EACirc/estream/ciphers/wg/wg.cpp \
    EACirc/estream/ciphers/tsc-4/tsc-4.cpp \
    EACirc/estream/ciphers/trivium/trivium.cpp \
    EACirc/estream/ciphers/sosemanuk/sosemanuk.cpp \
    EACirc/estream/ciphers/sfinks/sfinks.cpp \
    EACirc/estream/ciphers/salsa20/salsa20.cpp \
    EACirc/estream/ciphers/rabbit/rabbit.cpp \
    EACirc/estream/ciphers/py/py6.cpp \
    EACirc/estream/ciphers/pomaranch/pomaranch.cpp \
#    EACirc/estream/ciphers/polarbear/polar-bear.cpp \  # not implemented in EncryptorDecryptor
#    EACirc/estream/ciphers/polarbear/aescrypt.cpp \    # do not include!
#    EACirc/estream/ciphers/polarbear/aestab.cpp \      # do not include!
#    EACirc/estream/ciphers/polarbear/whirltab.cpp \    # do not include!
#    EACirc/estream/ciphers/nls/nlsref.cpp \            # not implemented in EncryptorDecryptor
#    EACirc/estream/ciphers/moustique/moustique.cpp \   # not implemented in EncryptorDecryptor
    EACirc/estream/ciphers/mir-1/mir-1.cpp \
    EACirc/estream/ciphers/mickey/mickey-128-v2.cpp \
    EACirc/estream/ciphers/mag/mag.cpp \
    EACirc/estream/ciphers/lex/lex.cpp \
    EACirc/estream/ciphers/hermes/hermes.cpp \
    EACirc/estream/ciphers/hc-128/hc-128.cpp \
    EACirc/estream/ciphers/grain/grain-v1.cpp \
    EACirc/estream/ciphers/fubuki/fubuki.cpp \
    EACirc/estream/ciphers/ffcsr/f-fcsr-h.cpp \
    EACirc/estream/ciphers/edon80/edon80.cpp \
    EACirc/estream/ciphers/dragon/dragon.cpp \
    EACirc/estream/ciphers/dragon/dragon-sboxes.cpp \
    EACirc/estream/ciphers/dicing/dicing-v2.cpp \
    EACirc/estream/ciphers/decim/decim-v2.cpp \
#    EACirc/estream/ciphers/decim/decim-128.c \         # do not include!
    EACirc/estream/ciphers/cryptmt/cryptmt-v3.cpp \     # not used
#    EACirc/estream/ciphers/cryptmt/altivec.cpp \       # do not include!
#    EACirc/estream/ciphers/cryptmt/sse2.cpp            # do not include!
    EACirc/estream/ciphers/achterbahn/achterbahn-128-80.cpp \
    EACirc/estream/ciphers/abc/abc-v3.cpp \             # not used
# === Libraries (redundant if using pre-compiled) ===
    EACirc/tinyXML/tinystr.cpp EACirc/tinyXML/tinyxml.cpp EACirc/tinyXML/tinyxmlerror.cpp EACirc/tinyXML/tinyxmlparser.cpp \
    EACirc/galib/GA1DArrayGenome.cpp EACirc/galib/GA1DBinStrGenome.cpp EACirc/galib/GA2DArrayGenome.cpp EACirc/galib/GA2DBinStrGenome.cpp \
    EACirc/galib/GA3DArrayGenome.cpp EACirc/galib/GA3DBinStrGenome.cpp EACirc/galib/GAAllele.cpp EACirc/galib/GABaseGA.cpp EACirc/galib/GABin2DecGenome.cpp \
    EACirc/galib/gabincvt.cpp EACirc/galib/GABinStr.cpp EACirc/galib/GADCrowdingGA.cpp EACirc/galib/GADemeGA.cpp EACirc/galib/gaerror.cpp EACirc/galib/GAGenome.cpp \
    EACirc/galib/GAIncGA.cpp EACirc/galib/GAList.cpp EACirc/galib/GAListBASE.cpp EACirc/galib/GAListGenome.cpp EACirc/galib/GAParameter.cpp EACirc/galib/GAPopulation.cpp \
    EACirc/galib/garandom.cpp EACirc/galib/GARealGenome.cpp EACirc/galib/GAScaling.cpp EACirc/galib/GASelector.cpp EACirc/galib/GASimpleGA.cpp EACirc/galib/GASStateGA.cpp \
    EACirc/galib/GAStatistics.cpp EACirc/galib/GAStringGenome.cpp EACirc/galib/GATree.cpp EACirc/galib/GATreeBASE.cpp EACirc/galib/GATreeGenome.cpp

HEADERS += \
# === Main project files ===
    EACirc/CircuitGenome.h \
    EACirc/CommonFnc.h \
    EACirc/XMLProcessor.h \
    EACirc/EACconstants.h \
    EACirc/globals_unused.h \
    EACirc/EACglobals.h \
    EACirc/Logger.h \
    EACirc/Evaluator.h \
    EACirc/status.h \
    EACirc/EACirc.h \
    EACirc/EAC_circuit.h \
    EACirc/test_vector_generator/ITestVectGener.h \
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
    EACirc/estream/EncryptorDecryptor.h \
# === eSTREAM cipher files ===
    EACirc/estream/estreamInterface.h \
    EACirc/estream/ciphers/ecrypt-config.h \
    EACirc/estream/ciphers/ecrypt-machine.h \
    EACirc/estream/ciphers/ecrypt-portable.h \
    EACirc/estream/ciphers/zk-crypt/ecrypt-sync.h \     # not used
    EACirc/estream/ciphers/zk-crypt/ZKdef.h \           # not used
    EACirc/estream/ciphers/zk-crypt/ZKengine.h \        # not used
    EACirc/estream/ciphers/yamb/ecrypt-sync.h \
    EACirc/estream/ciphers/wg/ecrypt-sync.h \
    EACirc/estream/ciphers/tsc-4/ecrypt-sync.h \
    EACirc/estream/ciphers/trivium/ecrypt-sync.h \
    EACirc/estream/ciphers/sosemanuk/ecrypt-sync.h \
    EACirc/estream/ciphers/sosemanuk/sosemanuk.h \
    EACirc/estream/ciphers/sfinks/ecrypt-sync.h \
    EACirc/estream/ciphers/salsa20/ecrypt-sync.h \
    EACirc/estream/ciphers/rabbit/ecrypt-sync.h \
    EACirc/estream/ciphers/py/ecrypt-sync.h \
    EACirc/estream/ciphers/pomaranch/ecrypt-sync.h \
    EACirc/estream/ciphers/pomaranch/pomaranch.h \
#    EACirc/estream/ciphers/polarbear/aes.h \           # not implemented in EncryptorDecryptor
#    EACirc/estream/ciphers/polarbear/aesopt.h \        # not implemented in EncryptorDecryptor
#    EACirc/estream/ciphers/polarbear/ecrypt-sync.h \   # not implemented in EncryptorDecryptor
#    EACirc/estream/ciphers/nls/ecrypt-sync.h \         # not implemented in EncryptorDecryptor
#    EACirc/estream/ciphers/nls/ecrypt-sync-ae.h \      # not implemented in EncryptorDecryptor
#    EACirc/estream/ciphers/nls/nls.h \                 # not implemented in EncryptorDecryptor
#    EACirc/estream/ciphers/nls/nlsmultab.h \           # not implemented in EncryptorDecryptor
#    EACirc/estream/ciphers/nls/nlssbox.h \             # not implemented in EncryptorDecryptor
#    EACirc/estream/ciphers/moustique/ecrypt-ssyn.h \   # not implemented in EncryptorDecryptor
    EACirc/estream/ciphers/mir-1/ecrypt-sync.h \
    EACirc/estream/ciphers/mir-1/mir1.h \
    EACirc/estream/ciphers/mickey/ecrypt-sync.h \
    EACirc/estream/ciphers/mag/ecrypt-sync.h \
    EACirc/estream/ciphers/mag/unrolliv.h \
    EACirc/estream/ciphers/mag/unrollmain.h \
    EACirc/estream/ciphers/lex/ecrypt-sync.h \
    EACirc/estream/ciphers/hermes/ecrypt-sync.h \
    EACirc/estream/ciphers/hc-128/ecrypt-sync.h \
    EACirc/estream/ciphers/grain/ecrypt-sync.h \
    EACirc/estream/ciphers/grain/grain-v1.h \
    EACirc/estream/ciphers/fubuki/ecrypt-sync.h \
    EACirc/estream/ciphers/ffcsr/ecrypt-sync.h \
    EACirc/estream/ciphers/ffcsr/ffcsrh-sync.h \
    EACirc/estream/ciphers/edon80/ecrypt-sync.h \
    EACirc/estream/ciphers/dragon/ecrypt-sync.h \
    EACirc/estream/ciphers/dicing/ecrypt-sync.h \
    EACirc/estream/ciphers/decim/decimv2.h \
#    EACirc/estream/ciphers/decim/decim-128.h        # do not include!
    EACirc/estream/ciphers/decim/ecrypt-sync.h \
    EACirc/estream/ciphers/cryptmt/ecrypt-sync.h \   # not used
    EACirc/estream/ciphers/cryptmt/params.h \        # not used
    EACirc/estream/ciphers/achterbahn/achterbahn.h \
    EACirc/estream/ciphers/achterbahn/ecrypt-sync.h \
    EACirc/estream/ciphers/abc/abc.h \               # not used
    EACirc/estream/ciphers/abc/abc-tables.h \        # not used
    EACirc/estream/ciphers/abc/ecrypt-sync.h \       # not used
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
    EACirc/test_vector_generator/EstreamTestVectGener.h

OTHER_FILES += \
    EACirc/config.xml
