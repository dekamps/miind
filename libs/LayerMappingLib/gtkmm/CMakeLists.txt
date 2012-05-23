project( ${MIIND_LIBRARY_PREFIX}layermapping_gtkmm )

link_directories(
    ${GTKMM_LIBRARY_DIRS}  )

include_directories(
    ${GTKMM_INCLUDE_DIRS}  )

set( layermapping_gtkmm_SRC
	ActivationArea.cpp
	LayerWidget.cpp
	LayerMappingApplet.cpp
	FeatureMapWidget.cpp
	ImageConverter.cpp
	NetworkWidget.cpp
	EnsembleWidget.cpp
 )

set( layermapping_gtkmm_HEADERS
	ActivationArea.h
	LayerWidget.h
	LayerMappingApplet.h
	FeatureMapWidget.h
	ImageConverter.h
	NetworkWidget.h
	EnsembleWidget.h
 )

add_library( ${MIIND_LIBRARY_PREFIX}layermapping_gtkmm ${LIB_TYPE} ${layermapping_gtkmm_SRC} )

target_link_libraries( ${MIIND_LIBRARY_PREFIX}layermapping_gtkmm
	${MIIND_LIBRARY_PREFIX}layermapping
	${GTKMM_LIBRARIES} )


install( FILES ${layermapping_gtkmm_HEADERS}
    DESTINATION include/miind/LayerMappingLib_gtkmm )

install( TARGETS ${MIIND_LIBRARY_PREFIX}layermapping_gtkmm
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )

