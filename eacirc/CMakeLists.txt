add_executable(eacirc main.cc)
target_link_libraries(eacirc core solvers)

target_sources(eacirc PRIVATE
    backend
    eacirc
    seed
    statistics
    streams/filestream
    version
    )

detect_version(version.h)
