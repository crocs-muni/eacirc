# === EACirc cmake file ===
include(${PROJECT_SOURCE_DIR}/cmake/detect_version.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/eacirc_project.cmake)


set(SOURCES
# === main project files ===
    Byte.h
    Main
    CommonFnc
    XMLProcessor
    Logger
    Status
    EACirc
    Finishers
    EACconstants.h
    EACglobals
    Finishers.h
    Version.h
    TestVectors.h
    cxx_utils.h
    metaheuristics
# === evaluators ===
    evaluators/IEvaluator
    evaluators/TopBitEvaluator
    evaluators/CategoriesEvaluator
    evaluators/HammingWeightEvaluator
    evaluators/FeatureEvaluator
# === circuit backend ===
    circuit/circuit
    circuit/interpreter
    circuit/operators
    circuit/settings
# === random generators ===
    generators/BiasRndGen
    generators/IRndGen
    generators/QuantumRndGen
    generators/MD5RndGen
    generators/md5
    generators/LUTRndGen
    generators/LUT
# === project interface ===
    projects/IProject
    projects/pregenerated_tv/PregeneratedTvProject
    projects/files/filesProject
    projects/files/filesConstants
# === testing using CATCH framework ===
    self_tests/Tests
    self_tests/TestConfigurator
)

set(OTHER_FILES
    config.xml
)

# === configure files ===
detect_version(Version.h)



# === source groups ===
source_group("cuda"            REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/cuda/.*")
source_group("evaluators"      REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/evaluators/.*")
source_group("generators"      REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/generators/.*")
source_group("projects"        REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/projects/.*")
source_group("projects\\pregenerated_tv" REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/projects/pregenerated_tv/.*")
source_group("projects\\files"           REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/projects/files/.*")
source_group("self_tests"      REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/self_tests/.*")



# === eacirc target ===
if(BUILD_CUDA)
    cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    cuda_add_executable(eacirc ${SOURCES} ${OTHER_FILES} OPTIONS -relaxed-constexpr)
    target_compile_definitions(eacirc PUBLIC USE_CUDA)
else()
    add_executable(eacirc ${SOURCES} ${OTHER_FILES})
endif()

target_include_directories(eacirc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECTS_DIR})
target_link_libraries(eacirc core minijson tinyXML catch)

# === build target to run directory ===
set_target_properties(eacirc PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${RUN_DIR})
add_custom_command(TARGET eacirc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_CURRENT_SOURCE_DIR}/config.xml -DDST=$<TARGET_FILE_DIR:eacirc>/config.xml -P ${CMAKE_SOURCE_DIR}/cmake/copy_if_not_exists.cmake
)


# === link projects ===
link_eacirc_project(caesar)
link_eacirc_project(estream)
link_eacirc_project(sha3)
