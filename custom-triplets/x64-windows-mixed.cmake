set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)

message( "Testing PORT ${PORT}." )

if(PORT MATCHES "boost")
	message ( " This boost lib should be dynamic.")
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