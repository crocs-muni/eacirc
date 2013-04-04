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
    EACirc/CommonFnc.cpp \
    EACirc/checker/Checker.cpp \
    EACirc/checker/CheckerMain.cpp \
    EACirc/Status.cpp \
    EACirc/Logger.cpp \
    EACirc/EACglobals.cpp \

# === evaluators ===
SOURCES += \
    EACirc/evaluators/IEvaluator.cpp \
    EACirc/evaluators/DistinguishTwoEvaluator.cpp \
    EACirc/evaluators/PredictBitCircuitEvaluator.cpp \
    EACirc/evaluators/PredictBitGroupParityCircuitEvaluator.cpp \
    EACirc/evaluators/PredictByteCircuitEvaluator.cpp \
    EACirc/evaluators/PredictBytesParityCircuitEvaluator.cpp \
    EACirc/evaluators/PredictHammingWeightCircuitEvaluator.cpp \

# === Main project files ===
HEADERS += \
    EACirc/checker/Checker.h \
    EACirc/checker/CheckerMain.h \
    EACirc/Status.h \
    EACirc/EACconstants.h \
    EACirc/Logger.h \
    EACirc/CommonFnc.h \
    EACirc/checker/EAC_circuit.h \
    EACirc/EACglobals.h \

# === evaluators ===
HEADERS += \
    EACirc/evaluators/IEvaluator.h \
    EACirc/evaluators/DistinguishTwoEvaluator.h \
    EACirc/evaluators/PredictBitCircuitEvaluator.h \
    EACirc/evaluators/PredictBitGroupParityCircuitEvaluator.h \
    EACirc/evaluators/PredictByteCircuitEvaluator.h \
    EACirc/evaluators/PredictBytesParityCircuitEvaluator.h \
    EACirc/evaluators/PredictHammingWeightCircuitEvaluator.h \
