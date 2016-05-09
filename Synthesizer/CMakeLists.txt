# === EACirc cmake file ===
include(${PROJECT_SOURCE_DIR}/cmake/detect_version.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/eacirc_project.cmake)

set(SOURCES
    Synthesizer
    LUT.cpp
    LUTRndGen.cpp
    MersenneTwisterRndGen.cpp
    PatternRndGen.cpp
    QuantumGen.cpp
)

set(HEADERS
    LUT.h
    LUTRndGen.h
    MersenneTwisterRndGen.h
    PatternRndGen.h
    QuantumGen.h
    RandGen.h
    Synthesizer.h
)


# === source groups ===

# === target ===
target_include_directories(eacirc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECTS_DIR})
set_target_properties(eacirc PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${RUN_DIR})
