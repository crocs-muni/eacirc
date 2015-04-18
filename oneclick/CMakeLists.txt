cmake_minimum_required(VERSION 3.0.2)

set(SOURCES
    src/ConfigParser.cpp
    src/FileGenerator.cpp
    src/main.cpp
    src/OneclickConstants.cpp
    src/ResultProcessor.cpp
    src/XMLproc.cpp
    # === project files ===
    ${PROJECTS_DIR}/estream/EstreamCiphers.cpp
    ${PROJECTS_DIR}/sha3/Sha3Functions.cpp
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
    ${PROJECTS_DIR}/estream/EstreamCiphers.h
    ${PROJECTS_DIR}/sha3/Sha3Functions.h
)


# === source groups ===
source_group(projects REGULAR_EXPRESSION "${CMAKE_SOURCE_DIR}/projects/.*")


# === oneclick target ===
add_executable(oneclick ${SOURCES} ${HEADERS})
target_include_directories(oneclick
    PRIVATE ${PROJECT_SOURCE_DIR}/eacirc
    PRIVATE ${PROJECTS_DIR}
)
target_link_libraries(oneclick tinyXML)


# === install ===
install(TARGETS oneclick DESTINATION ${RUN_DIR})