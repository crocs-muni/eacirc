add_executable(eacirc main.cc
    backend
    circuit/backend
    circuit/backend_impl
    circuit/circuit
    circuit/connectors
    circuit/functions
    circuit/genetics
    circuit/interpreter
    eacirc
    statistics
    )

target_link_libraries(eacirc eacirc-streams-lib eacirc-core solvers)

build_stream(eacirc estream)
build_stream(eacirc sha3)
build_stream(eacirc block)

