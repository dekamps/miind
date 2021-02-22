set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)

message( "Testing PORT ${PORT}." )

#if(PORT MATCHES "boost")
#	message ( " This boost lib should be dynamic.")
#    set(VCPKG_LIBRARY_LINKAGE dynamic)
#endif()

if(PORT MATCHES "boost-test")
	message ( " This boost-test lib should be dynamic.")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()

if(PORT MATCHES "boost-numeric-conversion")
	message ( " This boost-numeric-conversion lib should be dynamic.")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()

if(PORT MATCHES "boost-odeint")
	message ( " This boost-odeint lib should be dynamic.")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()

if(PORT MATCHES "boost-filesystem")
	message ( " This boost-filesystem lib should be dynamic.")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()

if(PORT MATCHES "boost-program-options")
	message ( " This boost-program-options lib should be dynamic.")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()

if(PORT MATCHES "boost-system")
	message ( " This boost-system lib should be dynamic.")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()

if(PORT MATCHES "boost-thread")
	message ( " This boost-thread lib should be dynamic.")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()

if(PORT MATCHES "boost-timer")
	message ( " This boost-timer lib should be dynamic.")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()

if(PORT MATCHES "boost-serialization")
	message ( " This boost-serialization lib should be dynamic.")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()

if(PORT MATCHES "boost-chrono")
	message ( " This boost-chrono lib should be dynamic.")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()

if(PORT MATCHES "boost-date-time")
	message ( " This boost-date-time lib should be dynamic.")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()

if(PORT MATCHES "boost-atomic")
	message ( " This boost-atomic lib should be dynamic.")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()

if(PORT MATCHES "boost-core")
	message ( " This boost-core lib should be dynamic.")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()

if(PORT MATCHES "boost-python")
	message ( " This boost-python lib should be dynamic.")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()

if(PORT STREQUAL "freeglut")
	message ( " This freeglut lib should be dynamic.")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()
if(PORT STREQUAL "glew")
	message ( " This glew lib should be dynamic.")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()
