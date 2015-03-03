# === tinyXML cmake file ===
cmake_minimum_required(VERSION 3.0.2)


set(SOURCES
    tinystr.cpp
    tinyxml.cpp
    tinyxmlerror.cpp
    tinyxmlparser.cpp
)

set(HEADERS
    tinystr.h
    tinyxml.h
)


add_library(tinyXML STATIC ${SOURCES} ${HEADERS} )

target_include_directories(tinyXML
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)
