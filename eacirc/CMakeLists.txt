add_executable(eacirc main.cc)
target_link_libraries(eacirc core)

target_sources(eacirc PRIVATE
	backend
    circuit/circuit
    circuit/circuit_backend
    circuit/functions
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
