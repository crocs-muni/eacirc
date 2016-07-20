add_executable(tests main.cc
        range
        range_iterator
        step_iterator
        variant
        settings
        )

target_link_libraries(tests catch core)
