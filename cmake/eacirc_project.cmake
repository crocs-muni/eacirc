cmake_minimum_required(VERSION 3.0.2)

function(add_eacirc_project NAME)
    string(TOUPPER ${NAME} UPPERCASE_NAME)

    add_library(${NAME} STATIC ${ARGN})
    target_compile_definitions(${NAME} INTERFACE -D${UPPERCASE_NAME})
    target_include_directories(${NAME} PRIVATE ${PROJECT_SOURCE_DIR}/eacirc)
endfunction()

function(link_eacirc_project NAME)
    string(TOUPPER ${NAME} UPPERCASE_NAME)

    if(BUILD_${UPPERCASE_NAME})
        target_link_libraries(eacirc ${NAME})
    endif()
endfunction()

function(include_eacirc_project NAME)
    string(TOUPPER ${NAME} UPPERCASE_NAME)

    option(BUILD_${UPPERCASE_NAME} "Build project ${NAME}" ON)

    if(BUILD_${UPPERCASE_NAME})
        add_subdirectory(${PROJECTS_DIR}/${NAME})
    endif()
endfunction()
