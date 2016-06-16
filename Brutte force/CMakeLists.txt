cmake_minimum_required(VERSION 3.3)

project(Brutte_force)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES_COMMON
        bit_array.h
        bithacks.h
        bithacks.cpp
        CommonFnc.cpp
        CommonFnc.h
        dynamic_bitset.h
        finisher.cpp
        finisher.h
        logger.cpp
        logger.h
        teestream.h
        Term.h
        TermGenerator.cpp
        TermGenerator.h
        )

set(SOURCE_FILES_KS
        ${SOURCE_FILES_COMMON}
        main.cpp)

set(SOURCE_FILES_BI
        ${SOURCE_FILES_COMMON}
        mainBi.h
        mainBi.cpp)

add_executable(Brutte_force ${SOURCE_FILES_KS})
add_executable(BrutteBi ${SOURCE_FILES_BI})
