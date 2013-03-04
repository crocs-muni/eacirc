TEMPLATE = app
CONFIG += console
CONFIG -= qt

unix{ QMAKE_CXX = g++-4.7 }

QMAKE_CXXFLAGS += -std=c++11 -Wall -Wextra #-Weffc++
INCLUDEPATH += ./EACirc

# === Main project files ===
SOURCES += \
    EACirc/Status.cpp \
    EACirc/Logger.cpp \
    EACirc/CommonFnc.cpp \
    EACirc/checker/Checker.cpp \
    EACirc/circuit_evaluator/DistinguishTwoEvaluator.cpp \
    EACirc/circuit_evaluator/ICircuitEvaluator.cpp \
    EACirc/circuit_evaluator/PredictAvalancheEvaluator.cpp \
    EACirc/circuit_evaluator/PredictBitCircuitEvaluator.cpp \
    EACirc/circuit_evaluator/PredictBitGroupParityCircuitEvaluator.cpp \
    EACirc/circuit_evaluator/PredictByteCircuitEvaluator.cpp \
    EACirc/circuit_evaluator/PredictBytesParityCircuitEvaluator.cpp \
    EACirc/circuit_evaluator/PredictHammingWeightCircuitEvaluator.cpp \
    EACirc/checker/CheckerMain.cpp \
    EACirc/EACglobals.cpp

HEADERS += \
    EACirc/Status.h \
    EACirc/EACconstants.h \
    EACirc/Logger.h \
    EACirc/CommonFnc.h \
    EACirc/checker/Checker.h \
    EACirc/circuit_evaluator/DistinguishTwoEvaluator.h \
    EACirc/circuit_evaluator/ICircuitEvaluator.h \
    EACirc/circuit_evaluator/PredictAvalancheEvaluator.h \
    EACirc/circuit_evaluator/PredictBitCircuitEvaluator.h \
    EACirc/circuit_evaluator/PredictBitGroupParityCircuitEvaluator.h \
    EACirc/circuit_evaluator/PredictByteCircuitEvaluator.h \
    EACirc/circuit_evaluator/PredictBytesParityCircuitEvaluator.h \
    EACirc/circuit_evaluator/PredictHammingWeightCircuitEvaluator.h \
    EACirc/EAC_circuit.h \
    EACirc/checker/CheckerMain.h \
    EACirc/EACglobals.h
