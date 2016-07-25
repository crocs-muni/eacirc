add_library(core STATIC
    ea-bitset
    ea-debug
    ea-iterators
    ea-json
    ea-logger
    ea-settings
    ea-stream
    ea-traits
    ea-utils
    ea-variant
    )

target_link_libraries(core json)
target_include_directories(core INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
