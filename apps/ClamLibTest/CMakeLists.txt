
if( WIN32 )
  find_package( WINGSL REQUIRED )
  if( NOT GSL_FOUND )
    message( FATAL_ERROR "WinGsl not found (module PopulistTest)." )
  endif( NOT GSL_FOUND )
  link_directories( ${GSL_LIBRARY_RELEASE_DIRS} )
else( WIN32 )
    include( RequireGSL )
    link_directories( ${GSL_LIBRARY_DIRS} )
endif( WIN32 )

include_directories( ${GSL_INCLUDE_DIRS} )

# this is done earlier
#find_package( Root )
#if( NOT ROOT_INCLUDE_DIRS AND NOT ROOT_LIBRARY_DIRS )
#    message( FATAL_ERROR "Root Library not found." )
#endif( NOT ROOT_INCLUDE_DIRS AND NOT ROOT_LIBRARY_DIRS )

include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )

include_directories( ../../libs/ClamLib )
include_directories( ../../libs/ConnectionismLib )
include_directories( ../../libs/DynamicLib )
include_directories( ../../libs/StructnetLib )

add_executable( ClamLibTest
    CLAM.cpp )

add_executable( Example2DSheet
	Example2DSheet.cpp )


target_link_libraries( Example2DSheet
    ${MIIND_LIBRARY_PREFIX}dynamic
    ${MIIND_LIBRARY_PREFIX}net
    ${MIIND_LIBRARY_PREFIX}numtools
    ${MIIND_LIBRARY_PREFIX}populist
    ${MIIND_LIBRARY_PREFIX}sparseimplementation
    ${MIIND_LIBRARY_PREFIX}clam
    ${MIIND_LIBRARY_PREFIX}util
    ${MIIND_LIBRARY_PREFIX}miind
    ${LIB_PREFIX}Core
    ${LIB_PREFIX}RIO
    ${LIB_PREFIX}Hist
    ${LIB_PREFIX}Tree
    ${LIB_PREFIX}Gpad
    ${LIB_PREFIX}MathCore
    ${LIB_PREFIX}MathMore
    gsl
    ${CBLAS_PREFIX}cblas )

target_link_libraries( ClamLibTest
    ${MIIND_LIBRARY_PREFIX}dynamic
    ${MIIND_LIBRARY_PREFIX}net
    ${MIIND_LIBRARY_PREFIX}numtools
    ${MIIND_LIBRARY_PREFIX}populist
    ${MIIND_LIBRARY_PREFIX}sparseimplementation
    ${MIIND_LIBRARY_PREFIX}clam
    ${MIIND_LIBRARY_PREFIX}util
    ${MIIND_LIBRARY_PREFIX}miind
    ${LIB_PREFIX}Core
    ${LIB_PREFIX}RIO
    ${LIB_PREFIX}Hist
    ${LIB_PREFIX}Tree
    ${LIB_PREFIX}Gpad
	${LIB_PREFIX}MathCore
	${LIB_PREFIX}MathMore
    gsl
    ${CBLAS_PREFIX}cblas )

