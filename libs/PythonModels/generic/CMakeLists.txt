project( ${MIIND_LIBRARY_PREFIX}pythonmodel )

link_directories( ${GSL_LIBRARY_DIRS} )
include( RequireGSL )
include_directories( ${GSL_INCLUDE_DIRS} )
include_directories( ../../ )
include_directories( ../../GeomLib )
include_directories( ../../TwoDLib )
set( LIB_TYPE SHARED )

IF (ENABLE_ROOT)
   # You need to tell CMake where to find the ROOT installation. This can be done in a number of ways:
   #   - ROOT built with classic configure/make use the provided $ROOTSYS/etc/cmake/FindROOT.cmake
   #   - ROOT built with CMake. Add in CMAKE_PREFIX_PATH the installation prefix for ROOT
   list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})

   #---Locate the ROOT package and defines a number of variables (e.g. ROOT_INCLUDE_DIRS)
   find_package(ROOT REQUIRED COMPONENTS RIO Net)

   #---Define useful ROOT functions and macros (e.g. ROOT_GENERATE_DICTIONARY)
   include(${ROOT_USE_FILE})

   include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

   # target_link_libraries(Event ${ROOT_LIBRARIES})
   add_definitions("-DWANTROOT")
ENDIF(ENABLE_ROOT)

#one below that MPILib is included in the paths they are needed for the camke config files
INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}/../
   ${CMAKE_CURRENT_BINARY_DIR}/../
)

set( PW_SRC
miind_generic.cpp
)

set( PW_HEADERS
)

set ( LIBLIST
    ${MIIND_LIBRARY_PREFIX}geom
    ${MIIND_LIBRARY_PREFIX}mpi
    ${MIIND_LIBRARY_PREFIX}twod
    ${MIIND_LIBRARY_PREFIX}numtools
    ${gsl_LINK_FLAGS}
    ${MPI_LIBRARIES}
    ${ROOT_LIBRARIES}
    ${Boost_LIBRARIES}
)

add_library( miindpython ${LIB_TYPE} ${PW_SRC} ${PW_HEADERS})

target_link_libraries( miindpython ${LIBLIST} -lpython2.7 -lboost_python)

install( FILES ${numtools_HEADERS}
    DESTINATION include/miind/PWlib/model )

install( TARGETS miindpython
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
