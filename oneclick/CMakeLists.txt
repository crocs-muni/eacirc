cmake_minimum_required(VERSION 3.0.2)


# === source files ===
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

# === header files ===
set(HEADERS
	src/AvgValPostPr.h
    src/ConfigParser.h
    src/dirent.h
    src/FileGenerator.h
    src/FileLogger.hpp
    src/FileSystem.h
    src/OneclickConstants.h
    src/PostProcessor.h
    src/PValuePostPr.h
    src/ResultProcessor.h
    src/Utils.h
    src/XMLproc.h
    # === project files ===
    ${PROJECTS_DIR}/estream/EstreamCiphers.h
    ${PROJECTS_DIR}/sha3/Sha3Functions.h
)

# === source groups ===
source_group(projects REGULAR_EXPRESSION "${PROJECTS_DIR}/.*")


# === oneclick target ===
add_executable(oneclick ${SOURCES} ${HEADERS})
target_include_directories(oneclick
    PRIVATE ${PROJECT_SOURCE_DIR}/eacirc
    PRIVATE ${PROJECTS_DIR}
)
target_link_libraries(oneclick tinyXML)


# === build target to oneclick run directory ===
set_target_properties(oneclick PROPERTIES RUNTIME_OUTPUT_DIRECTORY run)

add_custom_command(TARGET oneclick POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/run $<TARGET_FILE_DIR:oneclick>
)
