add_library(core STATIC
    base bitset iterators logger range teestream utils)

target_include_directories(core INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)
