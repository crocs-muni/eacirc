add_executable(eacirc main.cc)
target_link_libraries(eacirc core pcg)

target_sources(eacirc PRIVATE
    circuit/circuit
    circuit/genetics
    circuit/interpreter
    eacirc
    backend
    dataset
    seed
    solvers/local_search
    statistics
    version
    )

detect_version(version.h)
