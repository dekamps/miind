#
# Find the root library
# Volker Baier
# Adapted for use with MIIND
# MdK: 5-9-08, added lib for the root libraries in the windows branch

if( NOT WIN32 )
  SET( ROOT_POSSIBLE_ROOT_DIRS
      "${ROOT_ROOT_DIR}"
      "ENV{ROOT_ROOT_DIR}"
      root/
      "$ENV{ProgramFiles}\\root"
      /root/
      )
  FIND_PATH( ROOT_BINARY_DIRS
      NAMES
      root-config
      PATHS
      /root/bin/
      /opt/root/bin/
      /sw/root/bin/
      ${ROOT_POSSIBLE_ROOT_DIRS}/bin
      )

  FIND_PROGRAM( ROOT_CONFIG_SCRIPT root-config ${ROOT_BINARY_DIRS} )

  if( NOT ROOT_CONFIG_SCRIPT )
      message( FATAL_ERROR "Could not find root-config." )
  endif( NOT ROOT_CONFIG_SCRIPT )

  # if somone somhow has forgotten to set the ROOTSYS var...
  # we can do that on our own :-)
  if( NOT ROOTSYS )
      string( REGEX REPLACE "[/][b][i][n]" "" ROOTSYS ${ROOT_BINARY_DIRS} )
      set( ENV{ROOTSYS} ${ROOTSYS} )
  endif( NOT ROOTSYS )

  exec_program( ${ROOT_CONFIG_SCRIPT} ARGS "--incdir" OUTPUT_VARIABLE ROOT_INCLUDE_DIRS_TMP )
  set( ROOT_INCLUDE_DIRS ${ROOT_INCLUDE_DIRS_TMP} CACHE PATH "Include directory for root library." )

  exec_program( ${ROOT_CONFIG_SCRIPT} ARGS "--libdir" OUTPUT_VARIABLE ROOT_LIBRARY_DIRS_TMP )
  set( ROOT_LIBRARY_DIRS ${ROOT_LIBRARY_DIRS_TMP} CACHE PATH "Library directory for root library." )

  exec_program( ${ROOT_CONFIG_SCRIPT} ARGS "--libs" OUTPUT_VARIABLE ROOT_LDFLAGS_TMP )
  string( REGEX REPLACE "[-][L]([^ ])+" "" ROOT_LDFLAGS_TMP2 ${ROOT_LDFLAGS_TMP} )
  string( REGEX REPLACE "[-][l]" "" ROOT_LIBS ${ROOT_LDFLAGS_TMP2} )
  separate_arguments( ROOT_LIBS )

else( NOT WIN32 )
#
# Windows branch
#

  SET( ROOT_POSSIBLE_ROOT_DIRS
    "${ROOT_ROOT_DIR}"
    "ENV{ROOT_ROOT_DIR}"
    root
    "$ENV{ProgramFiles}\\root"
    /root
    )

  FIND_PATH( ROOT_ROOT_DIRS
    NAMES
    LICENSE
    PATHS
    ${ROOT_POSSIBLE_ROOT_DIRS}
    )

  FIND_PATH( ROOT_INCLUDE_DIRS
    NAMES
    RooRandom.h
    PATHS
    "${ROOT_ROOT_DIRS}/include"
    )

  FIND_PATH( ROOT_LIBRARY_DIRS
    NAMES
    libRooFit.lib
    PATHS
    "${ROOT_ROOT_DIRS}/lib"
    )
endif( NOT WIN32 )

if( ROOT_INCLUDE_DIRS AND ROOT_LIBRARY_DIRS )
    set( ROOT_FOUND TRUE )
endif( ROOT_INCLUDE_DIRS AND ROOT_LIBRARY_DIRS )

if( ROOT_FOUND )
    add_definitions( -DHAVE_LIBCORE )
    if( NOT ROOT_FIND_QUIETLY )
        message( STATUS "Found the ROOT library: ${ROOT_LIBRARY_DIRS}" )
    endif( NOT ROOT_FIND_QUIETLY )
    set( ROOTLIBS Core Hist)
else( ROOT_FOUND )
    if( ROOT_FIND_REQUIRED )
        message( FATAL_ERROR "Could not find the ROOT library." )
    endif( ROOT_FIND_REQUIRED )
endif( ROOT_FOUND )
