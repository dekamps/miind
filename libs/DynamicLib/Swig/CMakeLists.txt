#------------------------------------swig
if (SWIG_FOUND)
	INCLUDE_DIRECTORIES( .. )
	include_directories( ${GSL_INCLUDE_DIRS} )
	link_directories( ${GSL_LIBRARY_DIRS} )
	LINK_DIRECTORIES( ${ROOT_LIBRARY_DIRS} )
	SET_SOURCE_FILES_PROPERTIES(Dynamic.i PROPERTIES CPLUSPLUS ON)
	SWIG_ADD_MODULE( Dynamic python Dynamic.i )
	SWIG_LINK_LIBRARIES( Dynamic ${PYTHON_LIBRARIES} ${MIIND_LIBRARY_PREFIX}dynamic)
	ADD_DEFINITIONS( ${SWIG_COMPILE_OPTIONS} )
endif (SWIG_FOUND)
#------------------------------------swig
