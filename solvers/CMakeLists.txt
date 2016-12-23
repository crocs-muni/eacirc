add_library(solvers STATIC
    solvers
    individual
    local_search
    simulated_annealing
    solver_factory
    )

target_include_directories(solvers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    )

set_target_properties(solvers PROPERTIES
    LINKER_LANGUAGE CXX
    )

target_link_libraries(solvers
    core
    )
