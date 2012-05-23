#
# Find the windows gsl library
#
# lets guess some typical root dirs for wingsl

SET(GSL_POSSIBLE_ROOT_DIRS
    "${GSL_ROOT_DIR}"
    "$ENV{GSL_ROOT_DIR}"
    "C:/gsl"
    "$ENV{ProgramFiles}"
    )


FIND_PATH( GSL_ROOT_DIR
    NAMES
    "gsl"
    PATHS ${GSL_POSSIBLE_ROOT_DIRS}
    )

FIND_PATH( GSL_INCLUDE_DIRS
    NAMES
    gsl_mode.h
    PATHS ${GSL_ROOT_DIR}
    )

SET(GSL_LIBDIR_SUFFIXES
    lib
    )

FIND_PATH( GSL_LIBRARY_DIRS
    NAMES
    "Release"
    PATHS "${GSL_ROOT_DIR}/build.vc10/lib/Win32/"
    )

FIND_LIBRARY( GSL_LIBS
        NAMES gsl cblas
        PATHS "${GSL_LIBRARY_DIRS}/Release" PATH_SUFFIXES ${GSL_LIBDIR_SUFFIXES}
        )

if( GSL_INCLUDE_DIRS AND GSL_LIBRARY_DIRS )
    set( GSL_FOUND TRUE )
    set( ENV{GSL_INCLUDE_DIRS} ${GSL_INCLUDE_DIRS} )
    set( ENV{GSL_LIBRARY_DIRS} ${GSL_LIBRARY_DIRS} )
endif( GSL_INCLUDE_DIRS AND GSL_LIBRARY_DIRS )


if( GSL_FOUND )
    add_definitions( -DHAVE_GSL )
endif( GSL_FOUND )

