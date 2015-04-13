cmake_minimum_required(VERSION 3.0.2)


if(NOT EXISTS ${DST})
    message(STATUS "Installing: ${DST}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy ${SRC} ${DST})
else()
    message(STATUS "Skipping: ${DST}")
endif()
