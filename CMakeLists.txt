cmake_minimum_required(VERSION 3.4)
project(EACirc)

include(cmake/eacirc_project.cmake)

# === Set CXX flags ===
if(CMAKE_COMPILER_IS_GNUCXX OR ${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=c++11> -Wall -Wextra -Weffc++)
    add_compile_options(-fdiagnostics-color)

    # static build option is currently available only for linux
    option(STATIC_BUILD "Produce static buils" OFF)
elseif (MSVC AND (MSVC_VERSION GREATER 1700))
    # === nothing special yet ===
else()
    message(FATAL_ERROR "unsuported compiler: ${CMAKE_CXX_COMPILER}")
endif()

# === detect CUDA ===
find_package(CUDA)
if(CUDA_FOUND)
    option(BUILD_CUDA "Build CUDA code" ON)

    if(BUILD_CUDA)
        add_definitions(-DCUDA)
    endif()
endif()

# === Global variables ===
set(EACIRC_DIR   ${CMAKE_SOURCE_DIR}/eacirc)
set(PROJECTS_DIR ${CMAKE_SOURCE_DIR}/projects)
set(RUN_DIR      ${CMAKE_BINARY_DIR}/run)

# === static build? (applicable only on linux compilers) ===
if(STATIC_BUILD)
    SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    SET(BUILD_SHARED_LIBRARIES OFF)
    SET(CMAKE_EXE_LINKER_FLAGS "-static")
endif()

# === projects ===
include_eacirc_project(caesar)
include_eacirc_project(estream)
include_eacirc_project(sha3)

# === executables ===
add_subdirectory(eacirc)

# === 3rd party dependencies ===
add_subdirectory(libs/galib)
add_subdirectory(libs/tinyXML)
add_subdirectory(libs/catch)

# === testing ===
enable_testing()
if(BUILD_EACIRC)
    add_test(NAME self_test COMMAND eacirc -test WORKING_DIRECTORY $<TARGET_FILE_DIR:eacirc>)
    # this workaround adds custom target to display all outputs from failed test
    add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --build-config $<CONFIG>)
endif()
