set(CLAMDIR ../../libs/ClamLib )

include_directories( ${GSL_INCLUDE_DIRS} )
include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )

include_directories( ${CLAMDIR} )

add_executable( ROOTTest
    ROOTTest.cpp 
	dict.h
	LinkDef.h
	dict.cxx
	${CLAMDIR}/RootLayerDescription.cpp
	${CLAMDIR}/RootLayeredNetDescription.cpp
	${CMAKE_CURRENT_BINARY_DIR}/dict.cxx
)	




add_custom_command( OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dict.cxx
		    COMMAND rootcint -f ${CMAKE_CURRENT_BINARY_DIR}/dict.cxx -c ../${CLAMDIR}/RootLayerDescription.h ../${CLAMDIR}/RootLayeredNetDescription.h ${CMAKE_CURRENT_SOURCE_DIR}/LinkDef.h
)

target_link_libraries( ROOTTest
    ${LIB_PREFIX}Core
    ${LIB_PREFIX}Cint
    ${LIB_PREFIX}RIO
    ${LIB_PREFIX}Hist
    ${LIB_PREFIX}Tree
    ${LIB_PREFIX}Gpad
)


