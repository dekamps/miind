
include( UsePkgConfig )

#TODO do not restict to specific version
pkgconfig( gtk+-2.0 gtk_INCLUDE_DIR gtk_LINK_DIR gtk_LINK_FLAGS gtk_CFLAGS )

string(REGEX REPLACE "\n" " " gtk_INCLUDE_DIR "${gtk_INCLUDE_DIR}")
string(REGEX REPLACE "\n" " " gtk_CFLAGS "${gtk_CFLAGS}")
string(REGEX REPLACE "\n" " " gtk_LINK_FLAGS "${gtk_LINK_FLAGS}")

add_definitions( ${gtk_CFLAGS} )
link_directories( ${gtk_LINK_DIR} )

if( NOT gtk_CFLAGS OR NOT gtk_LINK_DIR )
    message( STATUS "Gtk not found, no GUI support." )
else( NOT gtk_CFLAGS OR NOT gtk_LINK_DIR )
    add_definitions( -DHAVE_GTK )
endif( NOT gtk_CFLAGS OR NOT gtk_LINK_DIR )


