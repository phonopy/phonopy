cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0007 NEW)

project(phonopy C)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_C_FLAGS_RELEASE "-Wall -O2")
set(CMAKE_C_FLAGS_DEBUG "-g -DTHMWARNING -DMATWARNING -DKPTWARNING")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

include(GNUInstallDirs)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# Version numbers
file(READ ${PROJECT_SOURCE_DIR}/phonopy/version.py version_file)
string(REGEX MATCH "__version__.*([0-9]+)[.]([0-9]+)[.]([0-9]+)" phonopy_version ${version_file})
set(phonopy_major_version ${CMAKE_MATCH_1})
set(phonopy_minor_version ${CMAKE_MATCH_2})
set(phonopy_micro_version ${CMAKE_MATCH_3})
set(serial "${phonopy_major_version}.${phonopy_minor_version}.${phonopy_micro_version}")
set(soserial "1")

# Source code
include_directories("${PROJECT_SOURCE_DIR}/c")
set(SOURCES
  ${PROJECT_SOURCE_DIR}/c/derivative_dynmat.c
  ${PROJECT_SOURCE_DIR}/c/dynmat.c
  ${PROJECT_SOURCE_DIR}/c/phonopy.c
  ${PROJECT_SOURCE_DIR}/c/rgrid.c
  ${PROJECT_SOURCE_DIR}/c/tetrahedron_method.c)

# Shared library
add_library(phpy SHARED ${SOURCES})

if(NOT MSVC)
  target_link_libraries(phpy m)
endif()
set_property(TARGET phpy PROPERTY VERSION ${serial})
set_property(TARGET phpy PROPERTY SOVERSION ${soserial})
install(TARGETS phpy LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Static link library
add_library(phpy_static STATIC ${SOURCES})
set_property(TARGET phpy_static PROPERTY VERSION ${serial})
set_property(TARGET phpy_static PROPERTY SOVERSION ${soserial})
set_property(TARGET phpy_static PROPERTY OUTPUT_NAME phpy)
install(TARGETS phpy_static ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Header file
install(FILES ${PROJECT_SOURCE_DIR}/c/phonopy.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
