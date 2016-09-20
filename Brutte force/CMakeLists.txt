cmake_minimum_required(VERSION 2.8)

project(Brutte_force)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES_COMMON
        base.h
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

set(SOURCE_FILES_ESTREAM
        DataGenerators/estream/EstreamInterface.h
        DataGenerators/estream/ciphers/ecrypt-config.h
        DataGenerators/estream/ciphers/ecrypt-machine.h
        DataGenerators/estream/ciphers/ecrypt-portable.h
        DataGenerators/estream/ciphers/decim/decimv2.h
        DataGenerators/estream/ciphers/decim/decim-v2.cpp
        DataGenerators/estream/ciphers/decim/ecrypt-sync.h
        )

set(SOURCE_FILES_SHA3
        DataGenerators/sha3/Sha3Constants.h
        DataGenerators/sha3/Sha3Interface.h
        DataGenerators/sha3/hash_functions/Tangle/Tangle_sha3.h
        DataGenerators/sha3/hash_functions/Tangle/Tangle_sha3.cpp
        DataGenerators/sha3/hash_functions/MD6/inttypes.h
        DataGenerators/sha3/hash_functions/MD6/md6.h
        DataGenerators/sha3/hash_functions/MD6/md6_compress.c
        DataGenerators/sha3/hash_functions/MD6/md6_mode.c
        DataGenerators/sha3/hash_functions/MD6/MD6_sha3.h
        DataGenerators/sha3/hash_functions/MD6/MD6_sha3.cpp
        DataGenerators/sha3/hash_functions/MD6/stdint.h
        )

set(SOURCE_DATA_SOURCES
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
        DataSource/DataSourceMD5.h
        DataGenerators/arcfour.cpp
        DataGenerators/arcfour.h
        DataSource/DataSourceRC4.cpp
        DataSource/DataSourceRC4.h
        DataSource/DataSourceRC4Column.cpp
        DataSource/DataSourceRC4Column.h
        DataSource/DataSourceSystemRandom.cpp
        DataSource/DataSourceSystemRandom.h
        DataSource/DataSourceDecim.cpp
        DataSource/DataSourceDecim.h
        DataSource/DataSourceDecimColumn.cpp
        DataSource/DataSourceDecimColumn.h
        DataSource/DataSourceSHA3.cpp DataSource/DataSourceSHA3.h)

set(SOURCE_FILES_KS
        ${SOURCE_FILES_COMMON}
        ${SOURCE_FILES_ESTREAM}
        ${SOURCE_FILES_SHA3}
        ${SOURCE_DATA_SOURCES}
        main.cpp
        )

set(SOURCE_FILES_MIX
        ${SOURCE_FILES_COMMON}
        ${SOURCE_FILES_ESTREAM}
        ${SOURCE_FILES_SHA3}
        ${SOURCE_DATA_SOURCES}
        mainMix.cpp
        )

set(SOURCE_FILES_BI
        ${SOURCE_FILES_COMMON}
        ${SOURCE_FILES_ESTREAM}
        ${SOURCE_FILES_SHA3}
        ${SOURCE_DATA_SOURCES}
        mainBi.h
        mainBi.cpp
        )

add_executable(Brutte_force ${SOURCE_FILES_KS})
add_executable(BrutteBi ${SOURCE_FILES_BI})
add_executable(BrutteMix ${SOURCE_FILES_MIX})

# BOOST
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS) #random
include_directories(${Boost_INCLUDE_DIRS})

# Linking
target_link_libraries(Brutte_force ${Boost_LIBRARIES})
target_link_libraries(BrutteBi ${Boost_LIBRARIES})
