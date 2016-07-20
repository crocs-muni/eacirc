add_library(metahuristics STATIC
        local_search
        )

set_target_properties(metahuristics PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(metahuristics INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
