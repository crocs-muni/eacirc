cmake_minimum_required(VERSION 3.4)
project(EACirc)

# === Set CXX flags ===
if(CMAKE_COMPILER_IS_GNUCXX OR ${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=c++11> -Wall -Wextra -Weffc++)
    add_compile_options(-fdiagnostics-color)
elseif (MSVC AND (MSVC_VERSION GREATER 1700))
    # nothing special yet
else()
    message(FATAL_ERROR "unsuported compiler: ${CMAKE_CXX_COMPILER}")
endif()

# === detect CUDA ===
find_package(CUDA)
if(CUDA_FOUND)
    option(ENABLE_CUDA "Enable CUDA acceleration" ON)
endif()

# === Global variables ===
set(EACIRC_DIR   ${CMAKE_SOURCE_DIR}/eacirc)
set(PROJECTS_DIR ${CMAKE_SOURCE_DIR}/projects)
set(RUN_DIR      ${CMAKE_BINARY_DIR}/run)

# === targets ===
add_subdirectory(core)
add_subdirectory(eacirc)

# === 3rd party dependencies ===
add_subdirectory(libs/catch)
add_subdirectory(libs/json)
