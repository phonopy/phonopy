cmake_minimum_required(VERSION 3.20)

if(NOT SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME phonopy)
endif()

if(PHONOPY_LIBS)
    project(phonopy LANGUAGES C)
    set(CMAKE_MACOSX_RPATH 1)
    set(CMAKE_C_FLAGS_RELEASE "-Wall -O2")
    set(CMAKE_C_FLAGS_DEBUG "-g -DTHMWARNING")
    include(GNUInstallDirs)

    # Version numbers
    file(READ ${PROJECT_SOURCE_DIR}/phonopy/version.py version_file)
    string(REGEX MATCH "__version__.*([0-9]+)[.]([0-9]+)[.]([0-9]+)"
        phono3py_version ${version_file})
    set(phonopy_major_version ${CMAKE_MATCH_1})
    set(phonopy_minor_version ${CMAKE_MATCH_2})
    set(phonopy_micro_version ${CMAKE_MATCH_3})
    set(serial
        "${phonopy_major_version}.${phonopy_minor_version}.${phonopy_micro_version}"
    )
    set(soserial "1")
else()
    project(${SKBUILD_PROJECT_NAME})
    set(DEV_MODULE Development.Module)
    find_package(
        Python 3.8
        REQUIRED COMPONENTS Interpreter ${DEV_MODULE}
        OPTIONAL_COMPONENTS Development.SABIModule
    )
endif()

cmake_policy(SET CMP0007 NEW)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE
        Release
        CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
        "MinSizeRel" "RelWithDebInfo")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_SYSTEM_PREFIX_PATH: ${CMAKE_SYSTEM_PREFIX_PATH}")

find_package(OpenMP)

if(OpenMP_FOUND)
    message(STATUS "OpenMP libs: ${OpenMP_C_LIBRARIES}")
    message(STATUS "OpenMP flags: ${OpenMP_C_FLAGS}")
endif()

set(SOURCES_PHONOPY
    ${PROJECT_SOURCE_DIR}/c/phonopy.c ${PROJECT_SOURCE_DIR}/c/dynmat.c
    ${PROJECT_SOURCE_DIR}/c/derivative_dynmat.c ${PROJECT_SOURCE_DIR}/c/rgrid.c
    ${PROJECT_SOURCE_DIR}/c/tetrahedron_method.c)

if(BUILD_SHARED_LIBS)
    # Shared library
    add_library(phonopy_libs SHARED ${SOURCES_PHONOPY})

    if(OpenMP_FOUND)
        target_link_libraries(
            phonopy_libs
            PRIVATE m
            PRIVATE OpenMP::OpenMP_C)
    else()
        target_link_libraries(phonopy_libs PRIVATE m)
    endif()

    target_compile_definitions(phonopy_libs PRIVATE THM_EPSILON=1e-10)
else()
    # Static link library
    add_library(phonopy_libs STATIC ${SOURCES_PHONOPY})

    if(OpenMP_FOUND)
        target_link_libraries(
            phonopy_libs
            PRIVATE m
            PRIVATE OpenMP::OpenMP_C)
    else()
        target_link_libraries(phonopy_libs PRIVATE m)
    endif()

    target_compile_definitions(phonopy_libs PRIVATE THM_EPSILON=1e-10)
endif()

if(PHONOPY_LIBS)
    install(FILES ${PROJECT_SOURCE_DIR}/c/phonopy.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    if(BUILD_SHARED_LIBS)
        install(TARGETS phonopy_libs LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
        set_property(TARGET phonopy_libs PROPERTY VERSION ${serial})
        set_property(TARGET phonopy_libs PROPERTY SOVERSION ${soserial})
    else()
        install(TARGETS phonopy_libs ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
        set_property(TARGET phonopy_libs PROPERTY VERSION ${serial})
        set_property(TARGET phonopy_libs PROPERTY SOVERSION ${soserial})
        set_property(TARGET phonopy_libs PROPERTY OUTPUT_NAME phonopy_libs)
    endif()
else()
    set_target_properties(phonopy_libs PROPERTIES POSITION_INDEPENDENT_CODE ON)

    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE NB_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
    find_package(nanobind CONFIG REQUIRED)
    nanobind_add_module(
        _phonopy
        STABLE_ABI
        ${PROJECT_SOURCE_DIR}/c/phonopy.h
        ${PROJECT_SOURCE_DIR}/c/_phonopy.cpp
    )

    if(OpenMP_FOUND)
        target_link_libraries(_phonopy PRIVATE phonopy_libs PRIVATE OpenMP::OpenMP_C)
    else()
        target_link_libraries(_phonopy PRIVATE phonopy_libs)
    endif()

    target_compile_definitions(_phonopy PRIVATE THM_EPSILON=1e-10)
    install(TARGETS _phonopy LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
endif()
