cmake_minimum_required(VERSION 3.0.2)
include(cmake/detect_version.cmake)


project(EACirc CXX C)


# === Set CXX flags ===
if(CMAKE_COMPILER_IS_GNUCXX OR ${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -Weffc++")
elseif (MSVC AND (MSVC_VERSION GREATER 1700))
    # === nothing special yet ===
else()
    message(FATAL_ERROR "unsuported compiler: ${CMAKE_CXX_COMPILER}")
endif ()


# === Enable profiling using gcc ===
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_GNUCC)
     option(ENABLE_PROFILING "Enable profiling" OFF)
    if(ENABLE_PROFILING)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -p")
    endif()
endif()




# === Define EACirc target name ===
set(EACIRC_TARGET EACirc)

#=== Define run directory ===
set(RUN_DIR ${CMAKE_BINARY_DIR}/run)


# === EACirc projects ===
option(BUILD_CAESAR "Enable building of Caesar" ON)
option(BUILD_ESTREAM "Enable building of Estream" ON)
option(BUILD_SHA3 "Enable building of SHA-3" ON)

add_subdirectory(projects/caesar)
add_subdirectory(projects/estream)
add_subdirectory(projects/sha3)


# === EACirc ===
add_subdirectory(EACirc)

# === Oneclick ===
option(BUILD_ONECLICK "Build Oneclick" OFF)
if(BUILD_ONECLICK)
    add_subdirectory(Oneclick)
endif()

# === 3rd party dependencies ===
add_subdirectory(libs/galib)
add_subdirectory(libs/tinyXML)
add_subdirectory(libs/catch)