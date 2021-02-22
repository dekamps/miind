if (WIN32 OR APPLE)
find_package( GSL )
else (WIN32 OR APPLE)
include( UsePkgConfig )

pkgconfig( gsl gsl_INCLUDE_DIR gsl_LINK_DIR gsl_LINK_FLAGS gsl_CFLAGS )

string(REGEX REPLACE "\n" " " gsl_INCLUDE_DIR "${gsl_INCLUDE_DIR}")
string(REGEX REPLACE "\n" " " gsl_CFLAGS "${gsl_CFLAGS}")
string(REGEX REPLACE "\n" " " gsl_LINK_FLAGS "${gsl_LINK_FLAGS}")

add_definitions( ${gsl_CFLAGS} )
link_directories( ${gsl_LINK_DIR} )

if( NOT gsl_CFLAGS OR NOT gsl_LINK_DIR )
    message( FATAL_ERROR "Gnu Scientific Library not found." )
endif( NOT gsl_CFLAGS OR NOT gsl_LINK_DIR )

endif (WIN32 OR APPLE)