##############################################################################
# include the header files
##############################################################################

#include the MPILib build dir for the config file
INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_BINARY_DIR}/../../libs/
)

include_directories( ../../libs/TwoDLib )
include_directories( ../../libs/MPILib )
include_directories( ../../libs/GeomLib )

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
    ${MIIND_LIBRARY_PREFIX}twod
    ${MIIND_LIBRARY_PREFIX}geom
    ${MIIND_LIBRARY_PREFIX}mpi
    ${MPI_LIBRARIES}
    ${ROOT_LIBRARIES}
    ${Boost_LIBRARIES}
    ${MIIND_LIBRARY_PREFIX}numtools
    ${gsl_LINK_FLAGS}
    gsl
    ${GLUT_LIBRARY}
)


#############################################################################
# executables
#############################################################################
add_executable(MatrixGenerator ConstructResetMapping.cpp	CorrectStrays.cpp		GenerateMatrix.cpp  MatrixGenerator.cpp TranslationObject.cpp)
add_executable(Bind Bind.cpp Bind.hpp)
target_link_libraries(Bind            ${LIBLIST})
target_link_libraries(MatrixGenerator ${LIBLIST})

set(APP_BASE_DIR share/miind/apps/MatrixGenerator/)

install( TARGETS Bind DESTINATION ${APP_BASE_DIR} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
install( TARGETS MatrixGenerator DESTINATION ${APP_BASE_DIR} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

if (SKBUILD) # in SKBUILD, only files in the bin directory are made exectuable.
install( TARGETS Bind DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
install( TARGETS MatrixGenerator DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

if (WIN32)
# Find vcruntime140.dll
find_path( VCRUNTIMEDIR vcruntime140.dll HINTS $ENV{Path} REQUIRED)
find_path( VCRUNTIME_1DIR vcruntime140_1.dll HINTS $ENV{Path} REQUIRED)
find_path( MSVCP140DIR msvcp140.dll HINTS $ENV{Path} REQUIRED)
message("Found VCRuntimes: ${VCRUNTIMEDIR} ${VCRUNTIME_1DIR} ${MSVCP140DIR} " )

# Now get the boost dlls - more difficult to get these because we only know where the libs are.
# Let's set up a few directories to search. Also add the location of the glut dll above just
# in case they're in the same place... cough VCPKG cough
set(MY_DEPENDENCY_PATHS ${GSL_LIBRARY_DIRS} ${GSL_LIBRARY_DIRS}/../bin/ ${VCRUNTIMEDIR} ${VCRUNTIME_1DIR} ${MSVCP140DIR}) 
cmake_policy(PUSH)
cmake_policy(SET CMP0026 OLD)
GET_TARGET_PROPERTY(OUTPUT_PATH MatrixGenerator LOCATION )
cmake_policy(POP)
# Transfer the value of ${OUTPUT_PATH} into the install script
install(CODE "set(MY_DEPENDENCY_PATHS_MG \"${MY_DEPENDENCY_PATHS}\")")
install(CODE "set(TARGET_LOCATION_MG \"${OUTPUT_PATH}\")")

install(CODE [[
  file(GET_RUNTIME_DEPENDENCIES
    LIBRARIES ${TARGET_LOCATION_MG}
    EXECUTABLES ${TARGET_LOCATION_MG}
    RESOLVED_DEPENDENCIES_VAR _r_deps
    UNRESOLVED_DEPENDENCIES_VAR _u_deps
	DIRECTORIES ${MY_DEPENDENCY_PATHS_MG}
  )
  
  #The GET_RUNTIME_DEPENDENCIES function has a filtering mechanism which would be better instead of the following filter
  foreach(_file ${_r_deps})
	if (NOT (${_file} MATCHES "gsl" OR ${_file} MATCHES "vcruntime140.dll" OR ${_file} MATCHES "VCRUNTIME140.dll" OR ${_file} MATCHES "vcruntime140_1.dll" OR ${_file} MATCHES "VCRUNTIME140_1.dll" OR ${_file} MATCHES "msvcp140.dll" OR ${_file} MATCHES "MSVCP140.dll"))
		continue()
	endif() 
	
    file(INSTALL
      DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
      TYPE SHARED_LIBRARY
      FILES "${_file}"
    )
  endforeach()
]])

endif(WIN32)

endif(SKBUILD)

