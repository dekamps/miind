##############################################################################
# include the header files
##############################################################################

#include the MPILib build dir for the config file
INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_BINARY_DIR}/../../libs/
)


include_directories( ../../libs/MPILib )

include_directories( ${GSL_INCLUDE_DIRS} )
link_directories( ${GSL_LIBRARY_DIRS} )
include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )

#############################################################################
# Libraries
#############################################################################

if (UNIX)
   ADD_DEFINITIONS("-fno-strict-aliasing" )
endif (UNIX)
set ( LIBLIST       
    ${MIIND_LIBRARY_PREFIX}mpi      
    ${MPI_LIBRARIES} 
    ${Boost_LIBRARIES}
    ${MIIND_LIBRARY_PREFIX}numtools
    ${gsl_LINK_FLAGS} 
	gsl
	${LIB_PREFIX}Core
    ${LIB_PREFIX}RIO
    ${LIB_PREFIX}Hist
    ${LIB_PREFIX}Tree
    ${LIB_PREFIX}Table
	${LIB_PREFIX}Matrix
	${LIB_PREFIX}MathCore
	${LIB_PREFIX}MathMore
    ${LIB_PREFIX}Gpad
	${LIB_PREFIX}Graf
	${LIB_PREFIX}Postscript
)


#############################################################################
# executables
#############################################################################
add_executable( miind-mpi
    miind-mpi.cpp )

target_link_libraries( miind-mpi ${LIBLIST} )

add_executable( two-population
    two-population.cpp )

target_link_libraries( two-population ${LIBLIST} )
