# === EACirc cmake file ===
cmake_minimum_required(VERSION 3.0.2)


set(SOURCES
# === main project files ===
    Main.cpp
    CommonFnc.cpp
    XMLProcessor.cpp
    Logger.cpp
    Status.cpp
    EACirc.cpp
    EACglobals.cpp
# === individual representation ===
    circuit/ICircuit.cpp
    circuit/ICircuitIO.cpp
# === circuit gate processing ===
    circuit/gate/GateCircuitIO.cpp
    circuit/gate/GateCircuit.cpp
    circuit/gate/GateInterpreter.cpp
    circuit/gate/GateCommonFunctions.cpp
    circuit/gate/GAGateCallbacks.cpp
# === polynomial circuits ===
    circuit/polynomial/PolynomialCircuit.cpp
    circuit/polynomial/PolynomialCircuitIO.cpp
    circuit/polynomial/Term.cpp
    circuit/polynomial/GAPolyCallbacks.cpp
    circuit/polynomial/PolyCommonFunctions.cpp
    circuit/polynomial/PolynomialInterpreter.cpp
# === evaluators ===
    evaluators/IEvaluator.cpp
    evaluators/TopBitEvaluator.cpp
    evaluators/CategoriesEvaluator.cpp
    evaluators/HammingWeightEvaluator.cpp
    evaluators/FeatureEvaluator.cpp
# === random generators ===
    generators/BiasRndGen.cpp
    generators/IRndGen.cpp
    generators/QuantumRndGen.cpp
    generators/MD5RndGen.cpp
    generators/md5.cpp
# === project interface ===
    projects/IProject.cpp
    projects/pregenerated_tv/PregeneratedTvProject.cpp
    projects/files/filesProject.cpp
# === testing using CATCH framework ===
    self_tests/Tests.cpp
    self_tests/TestConfigurator.cpp
)

set(HEADERS
# === main project files ===
    Main.h
    CommonFnc.h
    XMLProcessor.h
    EACconstants.h
    EACglobals.h
    Logger.h
    Status.h
    EACirc.h
    Version.h
# === individual representation ===
    circuit/ICircuit.h
    circuit/ICircuitIO.h
# === circuit gate processing ===
    circuit/gate/GateCircuit.h
    circuit/gate/GateCircuitIO.h
    circuit/gate/GateInterpreter.h
    circuit/gate/GateCommonFunctions.h
    circuit/gate/GAGateCallbacks.h
 # === polynomial circuits ===
    circuit/polynomial/GAPolyCallbacks.h
    circuit/polynomial/Term.h
    circuit/polynomial/PolynomialCircuit.h
    circuit/polynomial/PolynomialCircuitIO.h
    circuit/polynomial/PolyCommonFunctions.h
    circuit/polynomial/PolynomialInterpreter.h
# === evaluators ===
    evaluators/IEvaluator.h
    evaluators/TopBitEvaluator.h
    evaluators/CategoriesEvaluator.h
    evaluators/HammingWeightEvaluator.h
    evaluators/FeatureEvaluator.h
# === random generators ===
    generators/BiasRndGen.h
    generators/IRndGen.h
    generators/QuantumRndGen.h
    generators/MD5RndGen.h
    generators/md5.h
# === project interface ===
    projects/IProject.h
    projects/pregenerated_tv/PregeneratedTvProject.h
    projects/files/filesConstants.h
    projects/files/filesProject.h
# === testing using CATCH framework ===
    self_tests/TestConfigurator.h
)

set(OTHER_FILES
    config.xml
)


detect_version(Version.h)


add_executable(${EACIRC_TARGET} ${SOURCES} ${HEADERS} ${OTHER_FILES})

target_include_directories(${EACIRC_TARGET}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${PROJECTS_DIR}
)
target_link_libraries(${EACIRC_TARGET}
    galib
    tinyXML
    catch
)

if(BUILD_CAESAR)
    target_link_libraries(${EACIRC_TARGET} caesar)
    target_compile_definitions(${EACIRC_TARGET} PRIVATE -DCAESAR)
endif()
if(BUILD_ESTREAM)
    target_link_libraries(${EACIRC_TARGET} estream)
    target_compile_definitions(${EACIRC_TARGET} PRIVATE -DESTREAM)
endif()
if(BUILD_SHA3)
    target_link_libraries(${EACIRC_TARGET} sha3)
    target_compile_definitions(${EACIRC_TARGET} PRIVATE -DSHA3)
endif()


add_custom_command(TARGET ${EACIRC_TARGET} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory  ${RUN_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${EACIRC_TARGET}> ${RUN_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/config.xml ${RUN_DIR}
)