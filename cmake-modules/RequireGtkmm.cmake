
include( UsePkgConfig )

#TODO do not restict to specific version
pkgconfig( gtkmm-2.4 gtkmm_INCLUDE_DIR gtkmm_LINK_DIR gtkmm_LINK_FLAGS gtkmm_CFLAGS )

string(REGEX REPLACE "\n" " " gtkmm_INCLUDE_DIR "${gtkmm_INCLUDE_DIR}")
string(REGEX REPLACE "\n" " " gtkmm_CFLAGS "${gtkmm_CFLAGS}")
string(REGEX REPLACE "\n" " " gtkmm_LINK_FLAGS "${gtkmm_LINK_FLAGS}")

add_definitions( ${gtkmm_CFLAGS} )
link_directories( ${gtkmm_LINK_DIR} )

if( NOT gtkmm_CFLAGS OR NOT gtkmm_LINK_DIR OR NOT gtkmm_LINK_FLAGS )
    message( STATUS "Gtkmm not found, no GUI support." )
else( NOT gtkmm_CFLAGS OR NOT gtkmm_LINK_DIR OR NOT gtkmm_LINK_FLAGS )
    add_definitions( -DHAVE_GTKMM )
    message( STATUS "Gtkmm found, Building with gtkmm support." )
   
endif( NOT gtkmm_CFLAGS OR NOT gtkmm_LINK_DIR OR NOT gtkmm_LINK_FLAGS )


