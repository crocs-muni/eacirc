cmake_minimum_required(VERSION 3.4)
project(eacirc)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/eacirc-streams/eacirc-core/cmake)

include(build_stream)

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
add_subdirectory(eacirc-streams)

add_subdirectory(eacirc)

add_subdirectory(solvers)

add_custom_target(config SOURCES
        .travis.yml
        appveyor.yml
        config.json
        )
