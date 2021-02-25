link_directories( ${GSL_LIBRARY_DIRS} )
include( RequireGSL )
include_directories( ${GSL_INCLUDE_DIRS} )

set( LIB_TYPE SHARED )

find_package (Python3 COMPONENTS Development)
include_directories(${Python3_INCLUDE_DIRS})

message( "Found Python3. Include directory: ${Python3_INCLUDE_DIRS}.")

if (${Python3_INCLUDE_DIRS} STREQUAL "" )
message( "Counld not find Python.h. Try to find another.")
find_file(Python3_INCLUDE_DIRS "Python.h" HINTS "/opt/python/cp36-cp36m/" "/opt/python/cp38-cp38/" "/opt/python/cp36-cp36m/include" "/opt/python/cp38-cp38/include")
include_directories(${Python3_INCLUDE_DIRS})
message ("Found: ${Python3_INCLUDE_DIRS} ")
endif()

include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )

#one below that MPILib is included in the paths they are needed for the camke config files
INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}/../
   ${CMAKE_CURRENT_BINARY_DIR}/../
)

#############################################################################
# Libraries
#############################################################################
if (UNIX)
   ADD_DEFINITIONS("-fno-strict-aliasing" )
endif (UNIX)
set ( LIBLIST
    ${MIIND_LIBRARY_PREFIX}mpi
    ${Boost_LIBRARIES}
    ${MPI_LIBRARIES}
    ${MIIND_LIBRARY_PREFIX}numtools
    ${MIIND_LIBRARY_PREFIX}twod
    ${gsl_LINK_FLAGS}
    gsl
    ${GLUT_LIBRARY}
)

add_library( ${MIIND_LIBRARY_PREFIX}sim ${LIB_TYPE} MiindPython.cpp)

target_link_libraries( ${MIIND_LIBRARY_PREFIX}sim ${LIBLIST})

if (WIN32) # for windows, we want to set the ending to .pyd. In linux, the ending is just .so
SET_TARGET_PROPERTIES(${MIIND_LIBRARY_PREFIX}sim   PROPERTIES SUFFIX .pyd)
endif (WIN32)

install( TARGETS ${MIIND_LIBRARY_PREFIX}sim DESTINATION lib)
  
# In Windows, we want to redistribute the dependent dlls (Boost_CHRONO, Boost_FILESYSTEM, Boost_TIMER, and freeglut). 
# The following will find all dependencies of the output file and search for boost_chrono, boost_filesystem, boost_timer, and
# freeglut then install them too.
# I imagine there are a myriad ways that this could break but seriously, dependency stuff sucks so hard, and it
# took me all weekend just to get this to work.
if (WIN32)
## First lets get GLUT, it should be easy thanks to FindGLUT.
get_filename_component(glut_lib_basename "${GLUT_glut_LIBRARY}" NAME_WE)
install( FILES ${GLUT_RUNTIME_LIBRARY}/${glut_lib_basename}.dll DESTINATION bin)

# Now get the boost dlls - more difficult to get these because we only know where the libs are.
# Let's set up a few directories to search. Also add the location of the glut dll above just
# in case they're in the same place... cough VCPKG cough
set(MY_DEPENDENCY_PATHS ${Boost_LIBRARY_DIR_RELEASE} ${Boost_LIBRARY_DIR_RELEASE}/../bin/ ${GLUT_RUNTIME_LIBRARY} ${ADDITIONAL_DLL_DEPENDENCY_DIRS} ) 
cmake_policy(PUSH)
cmake_policy(SET CMP0026 OLD)
GET_TARGET_PROPERTY(OUTPUT_PATH ${MIIND_LIBRARY_PREFIX}sim LOCATION )
cmake_policy(POP)
# Transfer the value of ${OUTPUT_PATH} into the install script
install(CODE "set(MY_DEPENDENCY_PATHS \"${MY_DEPENDENCY_PATHS}\")")
install(CODE "set(TARGET_LOCATION \"${OUTPUT_PATH}\")")

install(CODE [[
  file(GET_RUNTIME_DEPENDENCIES
    LIBRARIES ${TARGET_LOCATION}
    EXECUTABLES ${TARGET_LOCATION}
    RESOLVED_DEPENDENCIES_VAR _r_deps
    UNRESOLVED_DEPENDENCIES_VAR _u_deps
	DIRECTORIES ${MY_DEPENDENCY_PATHS}
  )
  
  #The GET_RUNTIME_DEPENDENCIES function has a filtering mechanism which would be better instead of the following filter
  foreach(_file ${_r_deps})
	if (NOT (${_file} MATCHES "boost_chrono" OR ${_file} MATCHES "boost_filesystem" OR ${_file} MATCHES "boost_timer") )
		continue()
	endif() 
	
    file(INSTALL
      DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
      TYPE SHARED_LIBRARY
      FILES "${_file}"
    )
  endforeach()
]])
endif (WIN32)