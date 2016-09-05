add_executable(eacirc main.cc)
target_link_libraries(eacirc core solvers)

target_sources(eacirc PRIVATE
    backend
    circuit/backend
    circuit/backend_impl
    circuit/circuit
    circuit/connectors
    circuit/functions
    circuit/genetics
    circuit/interpreter
    eacirc
    seed
    statistics
    streams/filestream
    version
    )

detect_version(version.h)
