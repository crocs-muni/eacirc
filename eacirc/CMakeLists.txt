include(build_stream)
include(detect_version)

add_executable(eacirc main.cc
    backend
    circuit/backend
    circuit/backend_impl
    circuit/circuit
    circuit/connectors
    circuit/functions
    circuit/genetics
    circuit/interpreter
    evaluators/evaluator_factory
    eacirc
    seed
    statistics
    streams
    version
    )
target_link_libraries(eacirc core solvers)

detect_version(version.h)

build_stream(estream)
build_stream(sha3)
build_stream(block)
