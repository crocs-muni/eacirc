add_executable(eacirc main.cc)
target_link_libraries(eacirc core)

target_sources(eacirc PRIVATE
    circuit/circuit
    circuit/genetics
    circuit/interpreter
    eacirc
    backend
    dataset
    solvers/local_search
    statistics
    )
