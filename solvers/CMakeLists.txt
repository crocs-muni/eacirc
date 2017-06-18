add_library(solvers STATIC
    individual
    guided_local_search
    )

target_include_directories(solvers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    )

set_target_properties(solvers PROPERTIES
    LINKER_LANGUAGE CXX
    )

target_link_libraries(solvers
    eacirc-core
    )
