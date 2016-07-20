add_executable(eacirc main.cc)
target_link_libraries(eacirc core metahuristics)

target_sources(eacirc PRIVATE
    circuit/circuit
    circuit/circuit_factory.h
    circuit/genetics
    circuit/interpreter
    ea-eacirc
    ea-solver
    ea-backend
    )
