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
        DataGenerators/md5.h
        DataGenerators/md5.cpp
        DataGenerators/aes.h
        DataGenerators/aes.cpp
        DataSource/DataSource.cpp
        DataSource/DataSource.h
        DataSource/DataSourceFile.cpp
        DataSource/DataSourceFile.h
        DataSource/DataSourceAES.cpp
        DataSource/DataSourceAES.h
        DataSource/DataSourceMD5.cpp
        DataSource/DataSourceMD5.h)

set(SOURCE_FILES_KS
        ${SOURCE_FILES_COMMON}
        main.cpp)

set(SOURCE_FILES_BI
        ${SOURCE_FILES_COMMON}
        mainBi.h
        mainBi.cpp)

add_executable(Brutte_force ${SOURCE_FILES_KS})
add_executable(BrutteBi ${SOURCE_FILES_BI})

# BOOST
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS random)
include_directories(${Boost_INCLUDE_DIRS})

# Linking
target_link_libraries(Brutte_force ${Boost_LIBRARIES})
target_link_libraries(BrutteBi ${Boost_LIBRARIES})
