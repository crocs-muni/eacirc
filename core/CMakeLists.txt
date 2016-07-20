add_library(core STATIC
    ea-bitset
    ea-datastream
    ea-debug
    ea-iterators
    ea-logger
    ea-settings
    ea-traits
    ea-utils
    ea-variant
    )

target_include_directories(core INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
