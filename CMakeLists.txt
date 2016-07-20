cmake_minimum_required(VERSION 3.4)
project(eacirc)

# === Set CXX flags ===
if(CMAKE_COMPILER_IS_GNUCXX OR ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=c++14> -Wall -Wextra -Weffc++)
    add_compile_options(-fdiagnostics-color)
elseif (MSVC AND (MSVC_VERSION GREATER 1700))
    # nothing special yet
else()
    message(FATAL_ERROR "unsuported compiler id:${CMAKE_CXX_COMPILER_ID}, path: ${CMAKE_CXX_COMPILER}")
endif()

# === targets ===
add_subdirectory(core)
add_subdirectory(eacirc)
add_subdirectory(metaheuristics)
add_subdirectory(tests)

# === 3rd party dependencies ===
add_subdirectory(external/catch)
add_subdirectory(external/json)
