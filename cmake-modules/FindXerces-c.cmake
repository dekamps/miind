#
# Find the xercesc library
#

#if( NOT XERCESC_INCLUDE_DIRS )

set( XERCESC_INCLUDE_DIRS /usr/local/include CACHE PATH "Include path of Xerces-c library." )

if( NOT XERCESC_LIBRARY_DIRS )
    find_library( XERCESC_LIBRARY_DIRS
        NAMES xerces-c
        PATHS
        /usr/local/lib
        /usr/lib
        /lib
    )
    
    get_filename_component( XERCESC_LIBRARY_DIRS_COMPONENT "${XERCESC_LIBRARY_DIRS}" PATH)
    set( XERCESC_LIBRARY_DIRS ${XERCESC_LIBRARY_DIRS_COMPONENT} CACHE PATH "XERCESC library path" FORCE )
endif( NOT XERCESC_LIBRARY_DIRS )

if( XERCESC_INCLUDE_DIRS AND XERCESC_LIBRARY_DIRS )
    set( XERCESC_FOUND TRUE )
endif( XERCESC_INCLUDE_DIRS AND XERCESC_LIBRARY_DIRS )

set( XERCESC_LIBS xerces-c )

if( XERCESC_FOUND )
    if( NOT XERCESC_FIND_QUIETLY )
        message( STATUS "Found the XERCESC library: ${XERCESC_LIBRARY_DIRS}" )
    endif( NOT XERCESC_FIND_QUIETLY )
else( XERCESC_FOUND )
    if( XERCESC_FIND_REQUIRED )
        message( FATAL_ERROR "Could not find the XERCESC library." )
    endif( XERCESC_FIND_REQUIRED )
endif( XERCESC_FOUND )
