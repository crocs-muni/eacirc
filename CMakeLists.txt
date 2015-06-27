cmake_minimum_required(VERSION 3.0.2)
project(EACirc)

include(cmake/eacirc_project.cmake)

# === Set CXX flags ===
if(CMAKE_COMPILER_IS_GNUCXX OR ${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
    # === general flags ===
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")
    # === force coloured output (for Ninja and such) ===
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color")
    # === Supress warnings when building 3rd party libs on Linux ===
    set(SUPRESS_LIBS_WARNINGS "-Wall -Wextra -Weffc++" CACHE STRING "Supress warnings when building 3rd party libs")
elseif (MSVC AND (MSVC_VERSION GREATER 1700))
    # === nothing special yet ===
else()
    message(FATAL_ERROR "unsuported compiler: ${CMAKE_CXX_COMPILER}")
endif()

# === Enable profiling when using gcc ===
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_GNUCC)
     option(ENABLE_PROFILING "Enable profiling" OFF)
    if(ENABLE_PROFILING)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -p")
    endif()
endif()

# === Global variables ===
set(PROJECTS_DIR ${CMAKE_SOURCE_DIR}/projects)
set(RUN_DIR      ${CMAKE_BINARY_DIR}/run)

# === projects ===
include_eacirc_project(caesar)
include_eacirc_project(estream)
include_eacirc_project(sha3)
# === executables ===
add_subdirectory(eacirc)

option(BUILD_ONECLICK "Include oneclick in build" ON)
if(BUILD_ONECLICK)
    add_subdirectory(oneclick)
endif()
# === 3rd party dependencies ===
add_subdirectory(libs/galib)
add_subdirectory(libs/tinyXML)
add_subdirectory(libs/catch)

# === testing ===
enable_testing()
add_test(NAME test COMMAND eacirc -test WORKING_DIRECTORY $<TARGET_FILE_DIR:eacirc>)
