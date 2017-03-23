cmake_minimum_required(VERSION 3.4)
project(eacirc)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# === Set CXX flags ===
if(CMAKE_COMPILER_IS_GNUCXX OR ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=c++11> -Wall -Wextra)
    add_compile_options(-fdiagnostics-color)
elseif (MSVC AND (MSVC_VERSION GREATER 1700))
    # nothing special yet
else()
    message(FATAL_ERROR "unsuported compiler id:${CMAKE_CXX_COMPILER_ID}, path: ${CMAKE_CXX_COMPILER}")
endif()

# === targets ===
add_subdirectory(core)
add_subdirectory(eacirc)
add_subdirectory(solvers)

add_subdirectory(streams/estream)
add_subdirectory(streams/sha3)
add_subdirectory(streams/block)

add_subdirectory(external/json)
add_subdirectory(external/pcg-cpp-0.98)

add_custom_target(config SOURCES
    .travis.yml
    appveyor.yml
    config.json
    )
