target_sources(phonopy_lib PRIVATE
        derivative_dynmat.c
        dynmat.c
        phonopy.c
        rgrid.c
        tetrahedron_method.c
)
target_sources(_phonopy PRIVATE _phonopy.cpp)
target_include_directories(phonopy_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(phonopy_lib PRIVATE $<$<BOOL:${PHONOPY_USE_OMP}>:OpenMP::OpenMP_C>)
target_compile_definitions(phonopy_lib PRIVATE THM_EPSILON=1e-10)

target_link_libraries(_phonopy PRIVATE
        phonopy_lib
        $<$<BOOL:${PHONOPY_USE_OMP}>:OpenMP::OpenMP_CXX>
)
install(TARGETS phonopy_lib)
# Fix RPATH for unix systems
set_target_properties(_phonopy PROPERTIES
        INSTALL_RPATH "$<IF:$<BOOL:${APPLE}>,@loader_path,$ORIGIN>/${CMAKE_INSTALL_LIBDIR}"
)
# For windows systems we need to also install a copy of the dll files
install(TARGETS phonopy_lib RUNTIME DESTINATION .)
install(TARGETS _phonopy DESTINATION .)
