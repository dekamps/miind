
project( ${MIIND_LIBRARY_PREFIX}delayactivity )

if( WIN32 )
  find_package( WINGSL REQUIRED )
  if( NOT GSL_FOUND )
    message( FATAL_ERROR "WinGsl not found (module DelayActivityLib)." )
  endif( NOT GSL_FOUND )
  link_directories( ${GSL_LIBRARY_DIRS} )
  set( LIB_TYPE STATIC )
else( WIN32 )
  include( RequireGSL )
  set( LIB_TYPE SHARED )  
endif( WIN32 )

find_package( Root )
if( NOT ROOT_INCLUDE_DIRS AND NOT ROOT_LIBRARY_DIRS )
    message( FATAL_ERROR "Root Library not found." )
endif( NOT ROOT_INCLUDE_DIRS AND NOT ROOT_LIBRARY_DIRS )

include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )

set( delayactivity_SRC
    DelayActivityTest.cpp
    OrnsteinUhlenbeckAlgorithm.cpp
    Response.cpp )

set( delayactivity_HEADERS
    BasicDefinitions.h
    ConnectionSquaredProduct.h
    DelayActivityLib.h
    DelayActivityTest.h
    LocalDefinitions.h
    OrnsteinUhlenbeckAlgorithm.h
    OrnsteinUhlenbeckConnection.h
    OrnsteinUhlenbeckParameter.h
    Response.h
    ResponseParameter.h
    ResponseParameterAmit.h
    ResponseParameterBrunel.h
    TestDefinitions.h )

add_library( ${MIIND_LIBRARY_PREFIX}delayactivity ${LIB_TYPE} ${delayactivity_SRC} ${delayactivity_HEADERS})

target_link_libraries( ${MIIND_LIBRARY_PREFIX}delayactivity
    ${MIIND_LIBRARY_PREFIX}numtools
    ${MIIND_LIBRARY_PREFIX}net
    ${MIIND_LIBRARY_PREFIX}dynamic
    ${ROOT_LIBS}
    ${gsl_LINK_FLAGS} )

install( FILES ${delayactivity_HEADERS}
    DESTINATION include/miind/DelayActivityLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}delayactivity
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
