add_library(core STATIC
    builtins
    cmd
    dataset
    debug
    iterators
    json
    logger
    memory
    optional
    random
    statistics
    stream
    traits
    variant
    vec
    view
    )

target_include_directories(core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    )

set_target_properties(core PROPERTIES
    LINKER_LANGUAGE CXX
    )

target_link_libraries(core
    json
    pcg
    )
