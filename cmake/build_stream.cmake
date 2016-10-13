function(build_stream NAME)
    string(TOUPPER ${NAME} UPPERCASE_NAME)

    option(BUILD_${NAME} "Build stream ${NAME}" ON)

    if(BUILD_${NAME})
        target_link_libraries(eacirc ${NAME})
        target_compile_definitions(eacirc PRIVATE -DBUILD_${NAME})
    endif()
endfunction()
