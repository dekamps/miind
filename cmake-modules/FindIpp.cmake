#
# Find the ipp library
#

find_path( IPP_INCLUDE_DIRS
    ipp.h
    /usr/local/include
    /usr/include
    "$ENV{ProgramFiles}/Intel/IPP/5.1/ia32/include"
)

if( NOT IPP_LIBRARY_DIRS )
    find_library( IPP_LIBRARY_DIRS
        NAMES ippcore ippi ipps
        PATHS
        /usr/local/lib
        /usr/lib
        /lib
        "$ENV{ProgramFiles}/Intel/IPP/5.1/ia32/stublib"
    )
 
    get_filename_component( IPP_LIBRARY_DIRS_COMPONENT "${IPP_LIBRARY_DIRS}" PATH)
    set( IPP_LIBRARY_DIRS ${IPP_LIBRARY_DIRS_COMPONENT} CACHE PATH "IPP library path" FORCE )
endif( NOT IPP_LIBRARY_DIRS )

if( IPP_INCLUDE_DIRS AND IPP_LIBRARY_DIRS )
    set( IPP_FOUND TRUE )
endif( IPP_INCLUDE_DIRS AND IPP_LIBRARY_DIRS )

if ( NOT WIN32 )
  set( IPP_LIBS ippcore ippi ipps guide ippcc )
else( NOT WIN32 )
  set( IPP_LIBS ippcore ippi ipps ippc )
endif( NOT WIN32 )

if( IPP_FOUND )
    add_definitions( -DHAVE_IPP )
    if( NOT IPP_FIND_QUIETLY )
        message( STATUS "Found the IPP library: ${IPP_LIBRARY_DIRS}" )
    endif( NOT IPP_FIND_QUIETLY )
else( IPP_FOUND )
    if( IPP_FIND_REQUIRED )
        message( FATAL_ERROR "Could not find the IPP library." )
    endif( IPP_FIND_REQUIRED )
endif( IPP_FOUND )

