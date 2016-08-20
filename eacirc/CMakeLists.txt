add_executable(eacirc main.cc)
target_link_libraries(eacirc core)

target_sources(eacirc PRIVATE
    circuit/circuit
    circuit/genetics
    circuit/interpreter
    eacirc
    dataset
    seed
    solvers/criteria
    solvers/local_search
    solvers/solver
    statistics
    version
    )

detect_version(version.h)
