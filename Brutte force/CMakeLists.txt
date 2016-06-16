cmake_minimum_required(VERSION 3.3)


project(Brutte_force)



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")



set(SOURCE_FILES
    
    bit_array.h
    
    bithacks.h
    
    CommonFnc.cpp
    
    CommonFnc.h
    
    dynamic_bitset.h
    
    finisher.cpp
    
    finisher.h
    
    logger.cpp
    
    logger.h
    
    main.cpp
    
    teestream.h
    
    Term.h
)




add_executable(Brutte_force ${SOURCE_FILES})