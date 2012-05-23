
if( WIN32 )
  find_package( WINGSL REQUIRED )
  if( NOT GSL_FOUND )
    message( FATAL_ERROR "WinGsl not found." )
  endif( NOT GSL_FOUND )
  link_directories( ${GSL_LIBRARY_RELEASE_DIRS} )
else( WIN32 )
    link_directories( ${GSL_LIBRARY_DIRS} )
    include( RequireGSL )
endif( WIN32 )

include_directories( ${GSL_INCLUDE_DIRS} )

find_package( Root )
if( NOT ROOT_INCLUDE_DIRS AND NOT ROOT_LIBRARY_DIRS )
    message( FATAL_ERROR "Root Library not found." )
endif( NOT ROOT_INCLUDE_DIRS AND NOT ROOT_LIBRARY_DIRS )

include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )

include_directories(../../libs/AgentLib)
add_executable( trader 
    main.cpp )

target_link_libraries( trader
    ${MIIND_LIBRARY_PREFIX}dynamic
    ${MIIND_LIBRARY_PREFIX}agent
    ${MIIND_LIBRARY_PREFIX}miind
    ${MIIND_LIBRARY_PREFIX}net
    ${MIIND_LIBRARY_PREFIX}numtools
    ${MIIND_LIBRARY_PREFIX}populist
    ${MIIND_LIBRARY_PREFIX}sparseimplementation
    ${MIIND_LIBRARY_PREFIX}util
    ${MIIND_LIBRARY_PREFIX}connectionism
    ${MIIND_LIBRARY_PREFIX}structnet
    gsl
    cblas )
