
#------------------------------------swig
if (SWIG_FOUND)
        message("Kadaver")
	INCLUDE_DIRECTORIES( .. )
	include_directories( ${GSL_INCLUDE_DIRS} )
	link_directories( ${GSL_LIBRARY_DIRS} )
	LINK_DIRECTORIES( ${ROOT_LIBRARY_DIRS} )
	SET_SOURCE_FILES_PROPERTIES(Populist.i PROPERTIES CPLUSPLUS ON)
	SWIG_ADD_MODULE( Populist python Populist.i )
	SWIG_LINK_LIBRARIES( Populist ${PYTHON_LIBRARIES} ${MIIND_LIBRARY_PREFIX}populist ${MIIND_LIBRARY_PREFIX}miind)
	ADD_DEFINITIONS( ${SWIG_COMPILE_OPTIONS} )
endif (SWIG_FOUND)
#------------------------------------swig

