project( ${MIIND_LIBRARY_PREFIX}pythonmodule )

link_directories( ${GSL_LIBRARY_DIRS} )
include( RequireGSL )
include_directories( ${GSL_INCLUDE_DIRS} )
include_directories( ../../../libs/ )
include_directories( ../../../libs/GeomLib )
include_directories( ../../../libs/TwoDLib )
set( LIB_TYPE SHARED )

# TVB does not support MPI - as this Wilson Cowan shared library goes into the TVB source code,
# stick to MPI disabled. For other TVB-MIIND models, we can use MPI as we like.
set(ENABLE_MPI OFF)

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

set( TVB_WC_SRC
tvb_wilsoncowan.cpp
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

add_library( miindwc ${LIB_TYPE} ${TVB_WC_SRC} ${PW_HEADERS})

target_link_libraries( miindwc ${LIBLIST})
