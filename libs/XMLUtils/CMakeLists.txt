
project( ${MIIND_LIBRARY_PREFIX}xmlutils )

find_package( Xerces-c )

if( NOT XERCESC_FOUND )
    message( FATAL_ERROR "Xerces-c library not found." )
endif( NOT XERCESC_FOUND )

include_directories( ${XERCESC_INCLUDE_DIRS} )
link_directories( ${XERCESC_LIBRARY_DIRS} )

set( xmlutils_SRC
    CharString.cpp
    UtilDOMErrorHandler.cpp
    XMLFilter.cpp
    XMLHandler.cpp
    XMLStreamable.cpp
    XmlString.cpp )

set( xmlutils_HEADERS
    CharString.h
    UtilDOMErrorHandler.h
    XMLFilter.h
    XMLHandler.h
    XMLReader.h
    XMLReaderCode.h
    XMLStreamable.h
    XmlString.h
    XMLUtils.h )

add_library( ${MIIND_LIBRARY_PREFIX}xmlutils SHARED ${xmlutils_SRC} ${xmlutils_HEADERS})

target_link_libraries( ${MIIND_LIBRARY_PREFIX}xmlutils
    ${MIIND_LIBRARY_PREFIX}util
    ${XERCESC_LIBS} )

install( FILES ${xmlutils_HEADERS}
    DESTINATION include/miind/XMLUtils )

install( TARGETS ${MIIND_LIBRARY_PREFIX}xmlutils
    LIBRARY DESTINATION lib )

