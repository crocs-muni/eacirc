TEMPLATE = app
CONFIG += console
CONFIG -= qt

SUPPRESSED_WARNINGS = -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable \
                      -Wno-unused-function -Wno-unused-value
unix{ QMAKE_CXX = g++-4.7 }

QMAKE_LFLAGS_RELEASE += -static -static-libgcc -static-libstdc++
QMAKE_CXXFLAGS += -std=c++11 $$SUPPRESSED_WARNINGS -Wall -Wextra #-Weffc++
INCLUDEPATH += ./EACirc

# === Main project files ===
SOURCES += \
    EACirc/Status.cpp \
    EACirc/Logger.cpp \
    EACirc/CommonFnc.cpp \
    EACirc/checker/Checker.cpp \
    EACirc/evaluators/DistinguishTwoEvaluator.cpp \
    EACirc/evaluators/ICircuitEvaluator.cpp \
#    EACirc/evaluators/PredictAvalancheEvaluator.cpp \
    EACirc/evaluators/PredictBitCircuitEvaluator.cpp \
    EACirc/evaluators/PredictBitGroupParityCircuitEvaluator.cpp \
    EACirc/evaluators/PredictByteCircuitEvaluator.cpp \
    EACirc/evaluators/PredictBytesParityCircuitEvaluator.cpp \
    EACirc/evaluators/PredictHammingWeightCircuitEvaluator.cpp \
    EACirc/checker/CheckerMain.cpp \
    EACirc/EACglobals.cpp

HEADERS += \
    EACirc/Status.h \
    EACirc/EACconstants.h \
    EACirc/Logger.h \
    EACirc/CommonFnc.h \
    EACirc/checker/Checker.h \
    EACirc/evaluators/DistinguishTwoEvaluator.h \
    EACirc/evaluators/ICircuitEvaluator.h \
#    EACirc/evaluators/PredictAvalancheEvaluator.h \
    EACirc/evaluators/PredictBitCircuitEvaluator.h \
    EACirc/evaluators/PredictBitGroupParityCircuitEvaluator.h \
    EACirc/evaluators/PredictByteCircuitEvaluator.h \
    EACirc/evaluators/PredictBytesParityCircuitEvaluator.h \
    EACirc/evaluators/PredictHammingWeightCircuitEvaluator.h \
    EACirc/EAC_circuit.h \
    EACirc/checker/CheckerMain.h \
    EACirc/EACglobals.h
