#------------------------------------swig
if (SWIG_FOUND)
	INCLUDE_DIRECTORIES( .. )
	include_directories( ${GSL_INCLUDE_DIRS} )
	link_directories( ${GSL_LIBRARY_DIRS} )
	SET_SOURCE_FILES_PROPERTIES(Connectionism.i PROPERTIES CPLUSPLUS ON)
	SWIG_ADD_MODULE( Connectionism python Connectionism.i )
	SWIG_LINK_LIBRARIES( Connectionism ${PYTHON_LIBRARIES} ${MIIND_LIBRARY_PREFIX}connectionism)
	ADD_DEFINITIONS( ${SWIG_COMPILE_OPTIONS} )
endif (SWIG_FOUND)

#------------------------------------swig
