project (structnettest)


if( WIN32 )
  find_package( WINGSL REQUIRED )
  if( NOT GSL_FOUND )
    message( FATAL_ERROR "WinGsl not found (module structnettest)." )
  endif( NOT GSL_FOUND )
  link_directories( ${GSL_LIBRARY_DEBUG_DIRS} )
else( WIN32 )
    include( RequireGSL )
    link_directories( ${GSL_LIBRARY_DIRS} )
endif( WIN32 )

include_directories( ${GSL_INCLUDE_DIRS} )


add_executable( structnettest
    main.cpp )

target_link_libraries( structnettest
    ${MIIND_LIBRARY_PREFIX}numtools
    ${MIIND_LIBRARY_PREFIX}util
    ${MIIND_LIBRARY_PREFIX}net
    ${MIIND_LIBRARY_PREFIX}connectionism
    ${MIIND_LIBRARY_PREFIX}structnet
    gsl
    ${CBLAS_PREFIX}cblas )
