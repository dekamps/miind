#------------------------------------swig
if (SWIG_FOUND)
	INCLUDE_DIRECTORIES( .. )
	INCLUDE_DIRECTORIES( ${GTKMM_INCLUDE_DIRS} )
	SET_SOURCE_FILES_PROPERTIES(LayerMapping.i PROPERTIES CPLUSPLUS ON)
	#SET_SOURCE_FILES_PROPERTIES(LayerMapping.i PROPERTIES SWIG_FLAGS "-includeall")
	SWIG_ADD_MODULE( LayerMapping python LayerMapping.i )
	SWIG_LINK_LIBRARIES( LayerMapping ${PYTHON_LIBRARIES} ${MIIND_LIBRARY_PREFIX}layermapping)
	ADD_DEFINITIONS( ${SWIG_COMPILE_OPTIONS} )
endif (SWIG_FOUND)
#------------------------------------swig
