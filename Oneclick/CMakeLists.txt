cmake_minimum_required(VERSION 3.0.2)


set(SOURCES
    src/ConfigParser.cpp
    src/FileGenerator.cpp
    src/main.cpp
    src/OneclickConstants.cpp
    src/ResultProcessor.cpp
    src/XMLproc.cpp
    # === project files ===
    ${PROJECT_SOURCE_DIR}/projects/estream/EstreamCiphers.cpp
    ${PROJECT_SOURCE_DIR}/projects/sha3/Sha3Functions.cpp
)


set(HEADERS
    src/ConfigParser.h
    src/FileGenerator.h
    src/FileLogger.hpp
    src/FileSystem.h
    src/OneclickConstants.h
    src/ResultProcessor.h
    src/Utils.h
    src/XMLproc.h
    src/dirent.h
    # === project files ===
    ${PROJECT_SOURCE_DIR}/projects/estream/EstreamCiphers.h
    ${PROJECT_SOURCE_DIR}/projects/sha3/Sha3Functions.h
)


source_group(projects REGULAR_EXPRESSION "${CMAKE_SOURCE_DIR}/projects/.*")


add_executable(oneclick ${SOURCES} ${HEADERS})
target_include_directories(oneclick
    PRIVATE ${PROJECT_SOURCE_DIR}/EACirc
    PRIVATE ${PROJECT_SOURCE_DIR}/projects
)
target_link_libraries(oneclick tinyXML)