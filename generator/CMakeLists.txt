include(build_gen_stream)
include(detect_version)

add_executable(generator main.cc
    generator
    ../eacirc/seed
    ../eacirc/streams
    )
target_link_libraries(generator core)

build_gen_stream(estream)
build_gen_stream(sha3)
build_gen_stream(block)
