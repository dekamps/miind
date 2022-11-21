link_directories( ${GSL_LIBRARY_DIRS} )
include( RequireGSL )
include_directories( ${GSL_INCLUDE_DIRS} )

set( LIB_TYPE SHARED )

set( Python3_USE_STATIC_LIBS FALSE )

if (APPLE)
if (SKBUILD) # We're building using SKBUILD which means the correct python lib is passed as a flag! Hooray!
set(Python3_INCLUDE_DIRS ${PYTHON_INCLUDE_DIR} )
set(PYTHON_LIBRARIES ${PYTHON_LIBRARY})
set(Python3_LIBRARIES ${PYTHON_LIBRARY})
include_directories(${Python3_INCLUDE_DIRS})
else (SKBUILD)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
endif(SKBUILD)
else (APPLE)
if (WIN32 AND SKBUILD) # We're building using SKBUILD which means the correct python lib is passed as a flag! Hooray!
set(Python3_INCLUDE_DIRS ${PYTHON_INCLUDE_DIR} )
set(PYTHON_LIBRARIES ${PYTHON_LIBRARY})
set(Python3_LIBRARIES ${PYTHON_LIBRARY})
include_directories(${Python3_INCLUDE_DIRS})
else (WIN32 AND SKBUILD)
find_package (Python3 COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIR})
endif (WIN32 AND SKBUILD)
endif(APPLE)

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
    ${GLUT_LIBRARY}
    gsl
	${PYTHON_LIBRARIES}
)

message ( "Using Glut library: ${GLUT_LIBRARY} with ${GLUT_RUNTIME_LIBRARY_DIRS}")
message ( "Using python libraries: ${PYTHON_LIBRARIES} and ${Python3_LIBRARIES}")

#if (APPLE)
#set( LIBLIST ${LIBLIST} ${Python3_LIBRARIES} )
#endif (APPLE)


add_library( ${MIIND_LIBRARY_PREFIX}sim ${LIB_TYPE} MiindPython.cpp)

target_link_libraries( ${MIIND_LIBRARY_PREFIX}sim ${LIBLIST})

# On all platforms this should be called miindsim (not libmiindsim)
SET_TARGET_PROPERTIES(${MIIND_LIBRARY_PREFIX}sim PROPERTIES PREFIX "")

if (WIN32) # for windows, we want to set the ending to .pyd. In linux, the ending is just .so
SET_TARGET_PROPERTIES(${MIIND_LIBRARY_PREFIX}sim   PROPERTIES SUFFIX .pyd)
endif (WIN32)

if (APPLE) # for osx, we want to force the ending to .so instead of .dylib.
SET_TARGET_PROPERTIES(${MIIND_LIBRARY_PREFIX}sim   PROPERTIES SUFFIX .so)
endif (APPLE)

install( TARGETS ${MIIND_LIBRARY_PREFIX}sim DESTINATION lib)
  
# In Windows, we want to redistribute the dependent dlls (Boost_CHRONO, Boost_FILESYSTEM, Boost_TIMER, and freeglut). 
# The following will find all dependencies of the output file and search for boost_chrono, boost_filesystem, boost_timer, and
# freeglut then install them too.
# I imagine there are a myriad ways that this could break but seriously, dependency stuff sucks so hard, and it
# took me all weekend just to get this to work.
if (WIN32 AND SKBUILD)
## First lets get GLUT, it should be easy thanks to FindGLUT.
#get_filename_component(glut_lib_basename "${GLUT_glut_LIBRARY}" NAME_WE)
install( FILES ${GLUT_RUNTIME_LIBRARY} DESTINATION bin)

# Now get the boost dlls - more difficult to get these because we only know where the libs are.
# Let's set up a few directories to search. Also add the location of the glut dll above just
# in case they're in the same place... cough VCPKG cough
set(MY_DEPENDENCY_PATHS ${Boost_LIBRARY_DIR_RELEASE} ${Boost_LIBRARY_DIR_RELEASE}/../bin/ ${VCPKG_INSTALLED_DIR} ${ADDITIONAL_DLL_DEPENDENCY_DIRS} ) 
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
	if (NOT (${_file} MATCHES "boost_chrono"
 OR ${_file} MATCHES "boost_filesystem"
 OR ${_file} MATCHES "boost_timer" 
 OR ${_file} MATCHES "python"
 OR ${_file} MATCHES "glut") )
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
