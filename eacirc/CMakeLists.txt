# === EACirc cmake file ===
include(${PROJECT_SOURCE_DIR}/cmake/detect_version.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/eacirc_project.cmake)

set(SOURCES
# === main project files ===
    main
    backend
    eacirc
    finisher
    settings
# === legacy ===
    CommonFnc
    EACconstants.h
# === circuit backend ===
    circuit/circuit
    circuit/backend
    circuit/genotype
    circuit/interpreter
    circuit/operators
# === evaluators ===
    evaluators/categories
# === projects ===
    projects/file
# === evaluators ===
#    evaluators/IEvaluator
#    evaluators/TopBitEvaluator
#    evaluators/CategoriesEvaluator
#    evaluators/HammingWeightEvaluator
#    evaluators/FeatureEvaluator
# === solvers ===
    solvers/solver
    solvers/local_search
# === random generators ===
#    generators/BiasRndGen
#    generators/IRndGen
#    generators/QuantumRndGen
#    generators/MD5RndGen
#    generators/md5
#    generators/LUTRndGen
#    generators/LUT
# === project interface ===
#    projects/IProject
#    projects/pregenerated_tv/PregeneratedTvProject
#    projects/files/filesProject
#    projects/files/filesConstants
# === testing using CATCH framework ===
#    self_tests/Tests
#    self_tests/TestConfigurator
)

# === source groups ===
source_group("evaluators" REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/evaluators/.*")
source_group("generators" REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/generators/.*")
source_group("projects"   REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/projects/.*")

# === target ===
if(ENABLE_CUDA)
    cuda_add_executable(eacirc ${SOURCES})
    target_compile_definitions(eacirc PUBLIC CUDA)
else()
    add_executable(eacirc ${SOURCES})
endif()

find_package(yaml-cpp)
target_link_libraries(eacirc catch core yaml-cpp)
target_include_directories(eacirc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECTS_DIR})
set_target_properties(eacirc PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${RUN_DIR})
