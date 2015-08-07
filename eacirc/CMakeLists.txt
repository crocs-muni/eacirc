# === EACirc cmake file ===
cmake_minimum_required(VERSION 3.0.2)


include(${PROJECT_SOURCE_DIR}/cmake/detect_version.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/eacirc_project.cmake)


set(SOURCES
# === main project files ===
    Main.cpp
    CommonFnc.cpp
    XMLProcessor.cpp
    Logger.cpp
    Status.cpp
    EACirc.cpp
    Finishers.cpp
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
    circuit/gate/JVMSimulator.cpp
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
    Byte.h
    Main.h
    CommonFnc.h
    XMLProcessor.h
    EACconstants.h
    EACglobals.h
    Logger.h
    Status.h
    EACirc.h
    Finishers.h
    Version.h
    TestVectors.h
# === individual representation ===
    circuit/ICircuit.h
    circuit/ICircuitIO.h
# === circuit gate processing ===
    circuit/gate/GateCircuit.h
    circuit/gate/GateCircuitIO.h
    circuit/gate/GateInterpreter.h
    circuit/gate/GateCommonFunctions.h
    circuit/gate/GAGateCallbacks.h
    circuit/gate/JVMSimulator.h
 # === polynomial circuits ===
    circuit/polynomial/GAPolyCallbacks.h
    circuit/polynomial/Term.h
    circuit/polynomial/PolynomialCircuit.h
    circuit/polynomial/PolynomialCircuitIO.h
    circuit/polynomial/PolyCommonFunctions.h
    circuit/polynomial/PolynomialInterpreter.h
# === cuda ===
    cuda/base.h
    cuda/cudaSafeCall.h
    cuda/Host.h
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

set(CUDA_SOURCES    
# === gpu gate circuit ===
    circuit/gpu_gate/GaCallbacks.cpp
    circuit/gpu_gate/GpuGate.cpp
    circuit/gpu_gate/GpuTask.cpp
    circuit/gpu_gate/GpuTask.cu
)
set(CUDA_HEADERS
# === gpu gate circuit ===
    circuit/gpu_gate/Byte.h
    circuit/gpu_gate/GaCallbacks.h
    circuit/gpu_gate/GateCircuit.h
    circuit/gpu_gate/GateHelper.h
    circuit/gpu_gate/GateInterpreter.h
    circuit/gpu_gate/GpuGate.h
    circuit/gpu_gate/GpuTask.h
)

# === configure files ===
detect_version(Version.h)



# === source groups ===
source_group("circuit"         REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/circuit/.*")
source_group("circuit\\gate"            REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/circuit/gate/.*")
source_group("circuit\\gpu_gate"        REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/circuit/gpu_gate/.*")
source_group("circuit\\polynomial"      REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/circuit/polynomial/.*")
source_group("evaluators"      REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/evaluators/.*")
source_group("generators"      REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/generators/.*")
source_group("projects"        REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/projects/.*")
source_group("projects\\pregenerated_tv" REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/projects/pregenerated_tv/.*")
source_group("projects\\files"           REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/projects/files/.*")
source_group("self_tests"      REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/self_tests/.*")



# === eacirc target ===
if(BUILD_CUDA)
    cuda_include_dirs(${CMAKE_CURRENT_SOURCE_DIR})
    cuda_add_executable(eacirc ${SOURCES} ${HEADERS} ${OTHER_FILES} ${CUDA_SOURCES} ${CUDA_HEADERS} OPTIONS -std=c++11 -relaxed-constexpr)
    target_compile_definitions(eacirc PUBLIC USE_CUDA)
else()
    add_executable(eacirc ${SOURCES} ${HEADERS} ${OTHER_FILES})
endif()

target_include_directories(eacirc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECTS_DIR})
target_link_libraries(eacirc galib tinyXML catch)


# === build target to run directory ===
set_target_properties(eacirc PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${RUN_DIR})
add_custom_command(TARGET eacirc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_CURRENT_SOURCE_DIR}/config.xml -DDST=$<TARGET_FILE_DIR:eacirc>/config.xml -P ${CMAKE_SOURCE_DIR}/cmake/copy_if_not_exists.cmake
)


# === link projects ===
link_eacirc_project(caesar)
link_eacirc_project(estream)
link_eacirc_project(sha3)
