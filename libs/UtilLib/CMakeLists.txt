
project( ${MIIND_LIBRARY_PREFIX}util )

if ( WIN32 )
set (LIB_TYPE STATIC)
else( WIN32 ) 
set (LIB_TYPE SHARED)
endif (WIN32)

set( util_SRC
)
    
set( util_HEADERS
    BasicDefinitions.h
    LocalDefinitions.h
    UtilLib.h
 )

add_library( ${MIIND_LIBRARY_PREFIX}util ${LIB_TYPE} ${util_SRC} ${util_HEADERS})

install( FILES ${util_HEADERS}
    DESTINATION include/miind/UtilLib )


install( TARGETS ${MIIND_LIBRARY_PREFIX}util
    LIBRARY DESTINATION lib 
    ARCHIVE DESTINATION lib )
