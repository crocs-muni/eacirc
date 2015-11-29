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

# === options ===
option(BUILD_EACIRC "Include eacirc in build" ON)
option(BUILD_ONECLICK "Include oneclick in build" ON)

# === static build? (applicable only on linux compilers) ===
if(STATIC_BUILD)
    SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    SET(BUILD_SHARED_LIBRARIES OFF)
    SET(CMAKE_EXE_LINKER_FLAGS "-static")
endif()

# === projects ===
if(BUILD_EACIRC)
    include_eacirc_project(caesar)
    include_eacirc_project(estream)
    include_eacirc_project(sha3)
endif()
# === executables ===
if(BUILD_EACIRC)
    add_subdirectory(eacirc)
endif()

if(BUILD_ONECLICK)
    add_subdirectory(oneclick)
endif()
# === 3rd party dependencies ===
if(BUILD_EACIRC)
    add_subdirectory(libs/galib)
endif()
add_subdirectory(libs/tinyXML)
add_subdirectory(libs/catch)

# === testing ===
enable_testing()
if(BUILD_EACIRC)
    add_test(NAME self_test COMMAND eacirc -test WORKING_DIRECTORY $<TARGET_FILE_DIR:eacirc>)
    # this workaround adds custom target to display all outputs from failed test
    add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --build-config $<CONFIG>)
endif()
