if( GTKMM_FOUND ) #If gtkmm is found add layermapping_gui
  add_definitions( -DHAVE_GTKMM )	
  if ( NOT WIN32 )
    find_package( X11 REQUIRED)
    link_directories( ${GTKMM_LIBRARY_DIRS}
                      ${X11_LIBRARY_DIRS}
                      ${IMAGEMAGICKLIB_LIBRARY_DIRS}
                      ${IMAGEMAGICKLIB++_LIBRARY_DIRS}
                      ${IMAGEMAGICKLIBWAND_LIBRARY_DIRS} )
    include_directories( ${GTKMM_INCLUDE_DIRS}
                         ${X11_INCLUDE_DIR}
                         ${IMAGEMAGICKLIB_INCLUDE_DIRS})
    add_executable( layermapping_gui main_GUI.cpp )
    add_executable( layermapping_console main_console.cpp )
    add_executable( layermapping_simple_test simple_test.cpp )
    target_link_libraries( layermapping_gui
                           ${MIIND_LIBRARY_PREFIX}layermapping
                           ${MIIND_LIBRARY_PREFIX}layermapping_gtkmm
                           ${GTKMM_LIBRARIES}
                           ${IMAGEMAGICKLIB_LIBRARIES}
                           ${IMAGEMAGICKLIB++_LIBRARIES}
                           ${IMAGEMAGICKLIBWAND_LIBRARIES} )
    target_link_libraries( layermapping_console 
                           ${MIIND_LIBRARY_PREFIX}layermapping
                           ${GTKMM_LIBRARIES}
                           ${X11_LIBRARIES}
                           ${IMAGEMAGICKLIB_LIBRARIES}
                           ${IMAGEMAGICKLIB++_LIBRARIES}
                           ${IMAGEMAGICKLIBWAND_LIBRARIES} )
    target_link_libraries( layermapping_simple_test
                           ${GTKMM_LIBRARIES}
                           ${X11_LIBRARIES}
                           ${MIIND_LIBRARY_PREFIX}layermapping )
  else( NOT WIN32 )
    link_directories( ${GTKMM_LIBRARY_DIRS}
                      ${IMAGEMAGICKLIB_LIBRARY_DIRS}
                      ${IMAGEMAGICKLIB++_LIBRARY_DIRS}
                      ${IMAGEMAGICKLIBWAND_LIBRARY_DIRS} )
    include_directories( ${GTKMM_INCLUDE_DIRS}
                         ${IMAGEMAGICKLIB_INCLUDE_DIRS})
    add_executable( layermapping_gui main_GUI.cpp )
    add_executable( layermapping_console main_console.cpp )
    add_executable( layermapping_simple_test simple_test.cpp )
    target_link_libraries( layermapping_gui
                           ${MIIND_LIBRARY_PREFIX}layermapping
                           ${MIIND_LIBRARY_PREFIX}layermapping_gtkmm
                           ${GTKMM_LIBRARIES}
                           ${IMAGEMAGICKLIB_LIBRARIES}
                           ${IMAGEMAGICKLIB++_LIBRARIES}
                           ${IMAGEMAGICKLIBWAND_LIBRARIES} )
    target_link_libraries( layermapping_console
                           ${MIIND_LIBRARY_PREFIX}layermapping
                           ${GTKMM_LIBRARIES}
                           ${IMAGEMAGICKLIB_LIBRARIES}
                           ${IMAGEMAGICKLIB++_LIBRARIES}
                           ${IMAGEMAGICKLIBWAND_LIBRARIES} )
    target_link_libraries( layermapping_simple_test
                           ${GTKMM_LIBRARIES}
                           ${MIIND_LIBRARY_PREFIX}layermapping )
  endif( NOT WIN32 )
else( GTKMM_FOUND )
  link_directories( ${IMAGEMAGICKLIB_LIBRARY_DIRS} )
  include_directories( ${IMAGEMAGICKLIB_INCLUDE_DIRS})
  add_executable( layermapping_console main_console.cpp ) 
  add_executable( layermapping_simple_test simple_test.cpp )
  target_link_libraries( layermapping_console
                         ${MIIND_LIBRARY_PREFIX}layermapping
                         ${IMAGEMAGICKLIB_LIBRARIES}
                         ${IMAGEMAGICKLIB++_LIBRARIES}
                         ${IMAGEMAGICKLIBWAND_LIBRARIES} )
  target_link_libraries( layermapping_simple_test
                         ${MIIND_LIBRARY_PREFIX}layermapping )
endif( GTKMM_FOUND )
