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
    circuit/evaluators
    circuit/genotype
    circuit/interpreter
    circuit/module
    circuit/operators
# === evaluators ===
    evaluators/categories
# === projects ===
    projects/file
# === solvers ===
    solvers/solver
    solvers/local_search
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

target_link_libraries(eacirc catch core json)
target_include_directories(eacirc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECTS_DIR})
set_target_properties(eacirc PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${RUN_DIR})
