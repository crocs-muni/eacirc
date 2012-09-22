TEMPLATE = app
CONFIG += console
CONFIG -= qt

SUPPRESSED_WARNINGS = -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable \
                      -Wno-unused-function -Wno-unused-value
unix{
    QMAKE_CXX = g++-4.7
}
QMAKE_CXXFLAGS += -std=c++11 $$SUPPRESSED_WARNINGS

# to load GAlib and tinyXML as external libraries, do following:
# - comment the Libraries section in SOURCES and HEADERS
# - change includes in all files from "ga/header.h" to "header.h"
#   (same for tinyXML) - search for //libinclude in source files
# - uncomment following to lines (adding the libraries)
# QMAKE_CXXFLAGS += -isystem ../EACirc/ga -isystem ../EACirc/tinyXML
# LIBS += -ltinyXML -L../EACirc/tinyXML -lga -L../EACirc/ga

SOURCES += \
# === Main project files ===
    CircuitGenome.cpp \
    CommonFnc.cpp \
    EncryptorDecryptor.cpp \
    EstreamTestVectGener.cpp \
    Evaluator.cpp \
    ITestVectGener.cpp \
    status.cpp \
    EACirc.cpp \
    "Circuit Evaluator"/DistinguishTwoEvaluator.cpp \
    "Circuit Evaluator"/ICircuitEvaluator.cpp \
    "Circuit Evaluator"/PredictAvalancheEvaluator.cpp \
    "Circuit Evaluator"/PredictBitCircuitEvaluator.cpp \
    "Circuit Evaluator"/PredictBitGroupParityCircuitEvaluator.cpp \
    "Circuit Evaluator"/PredictByteCircuitEvaluator.cpp \
    "Circuit Evaluator"/PredictBytesParityCircuitEvaluator.cpp \
    "Circuit Evaluator"/PredictHammingWeightCircuitEvaluator.cpp \
    "Standalone testers"/TestDisctinctorCircuit.cpp \
    "Random Generator"/BiasRndGen.cpp \
    "Random Generator"/IRndGen.cpp \
    "Random Generator"/RndGen.cpp \
# === eSTREAM cipher files ===
    stream/zk-crypt/zk-crypt-v3.cpp \   # not used
    stream/yamb/yamb.cpp \
    stream/wg/wg.cpp \
    stream/tsc-4/tsc-4.cpp \
    stream/trivium/trivium.cpp \
    stream/sosemanuk/sosemanuk.cpp \
    stream/sfinks/sfinks.cpp \
    stream/salsa20/salsa20.cpp \
    stream/rabbit/rabbit.cpp \
    stream/py/py6.cpp \
    stream/pomaranch/pomaranch.cpp \
#    stream/polarbear/polar-bear.cpp \  # not implemented in EncryptorDecryptor
#    stream/polarbear/aescrypt.cpp \    # do not include!
#    stream/polarbear/aestab.cpp \      # do not include!
#    stream/polarbear/whirltab.cpp \    # do not include!
#    stream/nls/nlsref.cpp \            # not implemented in EncryptorDecryptor
#    stream/moustique/moustique.cpp \   # not implemented in EncryptorDecryptor
    stream/mir-1/mir-1.cpp \
    stream/mickey/mickey-128-v2.cpp \
    stream/mag/mag.cpp \
    stream/lex/lex.cpp \
    stream/hermes/hermes.cpp \
    stream/hc-128/hc-128.cpp \
    stream/grain/grain-v1.cpp \
    stream/fubuki/fubuki.cpp \
    stream/ffcsr/f-fcsr-h.cpp \
    stream/edon80/edon80.cpp \
    stream/dragon/dragon.cpp \
    stream/dragon/dragon-sboxes.cpp \
    stream/dicing/dicing-v2.cpp \
    stream/decim/decim-v2.cpp \
#    stream/decim/decim-128.c \         # do not include!
    stream/cryptmt/cryptmt-v3.cpp \     # not used
#    stream/cryptmt/altivec.cpp \       # do not include!
#    stream/cryptmt/sse2.cpp            # do not include!
    stream/achterbahn/achterbahn-128-80.cpp \
    stream/abc/abc-v3.cpp \             # not used
# === Libraries (redundant if using pre-compiled) ===
    tinyXML/tinystr.cpp tinyXML/tinyxml.cpp tinyXML/tinyxmlerror.cpp tinyXML/tinyxmlparser.cpp \
    ga/GA1DArrayGenome.cpp ga/GA1DBinStrGenome.cpp ga/GA2DArrayGenome.cpp ga/GA2DBinStrGenome.cpp \
    ga/GA3DArrayGenome.cpp ga/GA3DBinStrGenome.cpp ga/GAAllele.cpp ga/GABaseGA.cpp ga/GABin2DecGenome.cpp \
    ga/gabincvt.cpp ga/GABinStr.cpp ga/GADCrowdingGA.cpp ga/GADemeGA.cpp ga/gaerror.cpp ga/GAGenome.cpp \
    ga/GAIncGA.cpp ga/GAList.cpp ga/GAListBASE.cpp ga/GAListGenome.cpp ga/GAParameter.cpp ga/GAPopulation.cpp \
    ga/garandom.cpp ga/GARealGenome.cpp ga/GAScaling.cpp ga/GASelector.cpp ga/GASimpleGA.cpp ga/GASStateGA.cpp \
    ga/GAStatistics.cpp ga/GAStringGenome.cpp ga/GATree.cpp ga/GATreeBASE.cpp ga/GATreeGenome.cpp

HEADERS += \
# === Main project files ===
    CircuitGenome.h \
    CommonFnc.h \
    EncryptorDecryptor.h \
    estream-interface.h \
    EstreamVectGener.h \
    Evaluator.h \
    globals.h \
    ITestVectGener.h \
    SSconstants.h \
    SSGlobals.h \
    status.h \
    EACirc.h \
    EAC_circuit.h \
    "Circuit Evaluator"/DistinguishTwoEvaluator.h \
    "Circuit Evaluator"/ICircuitEvaluator.h \
    "Circuit Evaluator"/PredictAvalancheEvaluator.h \
    "Circuit Evaluator"/PredictBitCircuitEvaluator.h \
    "Circuit Evaluator"/PredictBitGroupParityCircuitEvaluator.h \
    "Circuit Evaluator"/PredictByteCircuitEvaluator.h \
    "Circuit Evaluator"/PredictBytesParityCircuitEvaluator.h \
    "Circuit Evaluator"/PredictHammingWeightCircuitEvaluator.h \
    "Standalone testers/TestDistinctorCircuit.h" \
    "Random Generator"/BiasRndGen.h \
    "Random Generator"/IRndGen.h \
    "Random Generator"/RndGen.h \
# === eSTREAM cipher files ===
    stream/ecrypt-config.h \
    stream/ecrypt-machine.h \
    stream/ecrypt-portable.h \
    stream/zk-crypt/ecrypt-sync.h \     # not used
    stream/zk-crypt/ZKdef.h \           # not used
    stream/zk-crypt/ZKengine.h \        # not used
    stream/yamb/ecrypt-sync.h \
    stream/wg/ecrypt-sync.h \
    stream/tsc-4/ecrypt-sync.h \
    stream/trivium/ecrypt-sync.h \
    stream/sosemanuk/ecrypt-sync.h \
    stream/sosemanuk/sosemanuk.h \
    stream/sfinks/ecrypt-sync.h \
    stream/salsa20/ecrypt-sync.h \
    stream/rabbit/ecrypt-sync.h \
    stream/py/ecrypt-sync.h \
    stream/pomaranch/ecrypt-sync.h \
    stream/pomaranch/pomaranch.h \
#    stream/polarbear/aes.h \           # not implemented in EncryptorDecryptor
#    stream/polarbear/aesopt.h \        # not implemented in EncryptorDecryptor
#    stream/polarbear/ecrypt-sync.h \   # not implemented in EncryptorDecryptor
#    stream/nls/ecrypt-sync.h \         # not implemented in EncryptorDecryptor
#    stream/nls/ecrypt-sync-ae.h \      # not implemented in EncryptorDecryptor
#    stream/nls/nls.h \                 # not implemented in EncryptorDecryptor
#    stream/nls/nlsmultab.h \           # not implemented in EncryptorDecryptor
#    stream/nls/nlssbox.h \             # not implemented in EncryptorDecryptor
#    stream/moustique/ecrypt-ssyn.h \   # not implemented in EncryptorDecryptor
    stream/mir-1/ecrypt-sync.h \
    stream/mir-1/mir1.h \
    stream/mickey/ecrypt-sync.h \
    stream/mag/ecrypt-sync.h \
    stream/mag/unrolliv.h \
    stream/mag/unrollmain.h \
    stream/lex/ecrypt-sync.h \
    stream/hermes/ecrypt-sync.h \
    stream/hc-128/ecrypt-sync.h \
    stream/grain/ecrypt-sync.h \
    stream/grain/grain-v1.h \
    stream/fubuki/ecrypt-sync.h \
    stream/ffcsr/ecrypt-sync.h \
    stream/ffcsr/ffcsrh-sync.h \
    stream/edon80/ecrypt-sync.h \
    stream/dragon/ecrypt-sync.h \
    stream/dicing/ecrypt-sync.h \
    stream/decim/decimv2.h \
#    stream/decim/decim-128.h        # do not include!
    stream/decim/ecrypt-sync.h \
    stream/cryptmt/ecrypt-sync.h \   # not used
    stream/cryptmt/params.h \        # not used
    stream/achterbahn/achterbahn.h \
    stream/achterbahn/ecrypt-sync.h \
    stream/abc/abc.h \               # not used
    stream/abc/abc-tables.h \        # not used
    stream/abc/ecrypt-sync.h \       # not used
# === Libraries (redundant if using pre-compiled) ===
    tinyXML/tinystr.h tinyXML/tinyxml.h \
    ga/GA1DArrayGenome.h ga/ga.h ga/GA1DBinStrGenome.h ga/GA2DArrayGenome.h ga/GA2DBinStrGenome.h \
    ga/GA3DArrayGenome.h ga/GA3DBinStrGenome.h ga/GAAllele.h ga/GAArray.h ga/GABaseGA.h \
    ga/GABin2DecGenome.h ga/gabincvt.h ga/GABinStr.h ga/gaconfig.h ga/GADCrowdingGA.h ga/GADemeGA.h \
    ga/gaerror.h ga/GAEvalData.h ga/GAGenome.h ga/gaid.h ga/GAIncGA.h ga/GAList.h ga/GAListBASE.h \
    ga/GAListGenome.h ga/GAMask.h ga/GANode.h ga/GAParameter.h ga/GAPopulation.h ga/garandom.h \
    ga/GARealGenome.h ga/GAScaling.h ga/GASelector.h ga/GASimpleGA.h ga/GASStateGA.h ga/GAStatistics.h \
    ga/GAStringGenome.h ga/GATree.h ga/GATreeBASE.h ga/GATreeGenome.h ga/gatypes.h ga/gaversion.h \
    ga/std_stream.h

OTHER_FILES += \
    config.xml
