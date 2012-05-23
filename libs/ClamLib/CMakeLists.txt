project( ${MIIND_LIBRARY_PREFIX}clam )

if( WIN32 )
  find_package( WINGSL REQUIRED )
  if( NOT GSL_FOUND )
    message( FATAL_ERROR "WinGsl not found (module Populist)." )
  endif( NOT GSL_FOUND )
  link_directories( ${GSL_LIBRARY_RELEASE_DIRS} )
  set (LIB_TYPE STATIC)
else( WIN32 )
  include( RequireGSL )
  link_directories( ${GSL_LIBRARY_DIRS} )
  if ( APPLE )             #This is not necessary any more under leopard
    set( LIB_TYPE STATIC ) #but it makes no differnece
  else( APPLE )
    set( LIB_TYPE SHARED )
  endif( APPLE )
endif( WIN32 )

include_directories( ${GSL_INCLUDE_DIRS} )

find_package( Root )
if( NOT ROOT_FOUND )
    message( FATAL_ERROR "Root Library not found. Is essential for ClamLib." )
endif( NOT ROOT_FOUND )

include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )

set( clam_SRC
AbstractCircuitCreator.cpp
AddTrainedNetToDynamicNetwork.cpp
AnalysisMacro.cpp
CircuitDescription.cpp
CircuitInfo.cpp
CircuitNodeRole.cpp
ClamLibTest.cpp
ConfigurableCreator.cpp
CreateDevelopingNetworks.cpp
CreatorStockObjects.cpp
DynamicSubNetwork.cpp
ForwardSubNetworkIterator.cpp
HomeostaticSmooth.cpp
Id.cpp
IndexWeight.cpp
InputCircuitCreator.cpp
InverseInputRate.cpp
JOCNPattern.cpp
Load.cpp
Macro.cpp
NetId.cpp
ParseNeatoConfig.cpp
PerceptronCircuitCreator.cpp
PerceptronConfigurableCreator.cpp
ReverseSubNetworkIterator.cpp
RootLayerDescription.cpp
RootLayeredNetDescription.cpp
SemiSigmoid.cpp
SimpleCircuitCreator.cpp
SimulationInfoBlock.cpp
SimulationInfoBlockVector.cpp
SimulationResult.cpp
SimulationOrganizer.cpp
TrainedNet.cpp
WeightList.cpp)

set( clam_HEADERS
AbstractCircuitCreator.h
AddTrainedNetToDynamicNetwork.h
BasicDefinitions.h
CircuitDescription.h
CircuitInfo.h
CircuitNodeRole.h
ClamLibException.h
ClamLib.h
ClamLibPositions.h
ClamLibTest.h
ConfigurableCreator.h
CreateDevelopingNetworks.h
CreatorStockObjects.h
DynamicSubNetwork.h
ForwardSubNetworkIterator.h
HomeostaticSmooth.h
Id.h
inclvector.h
IndexWeight.h
InputCircuitCreator.h
InverseInputRate.h
JocnPattern.h
LinkDef.h
LocalDefinitions.h
NetId.h
ParseNeatoConfig.h
PerceptronCircuitCreator.h
PerceptronConfigurableCreator.h
PerceptronOffsets.h
ReverseSubNetworkIterator.h
RootConversions.h
RootLayerDescription.h
RootLayeredNetDescription.h
RootNetCatalog.h
SemiSigmoid.h
SimpleCircuitCreator.h
SimulationInfoBlock.h
SimulationInfoBlockVector.h
SimulationOrganizer.h
SpatPosFromPhysPos.h
TestDefinitions.h
ToRootLayerDescription.h
ToRootLayeredNetDescription.h
TrainedNet.h
WeightedLink.h
WeightList.h
)

add_custom_command( OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dict.cxx
		    COMMAND rootcint -f # removed path (MdK 14-06-2011), it's now assumed that root/bin is in the systems path variable
${CMAKE_CURRENT_BINARY_DIR}/dict.cxx -c ${CMAKE_CURRENT_SOURCE_DIR}/Id.h
${CMAKE_CURRENT_SOURCE_DIR}/CircuitInfo.h
${CMAKE_CURRENT_SOURCE_DIR}/CircuitDescription.h
${CMAKE_CURRENT_SOURCE_DIR}/CircuitNodeRole.h
${CMAKE_CURRENT_SOURCE_DIR}/IndexWeight.h 
${CMAKE_CURRENT_SOURCE_DIR}/InputOutputPair.h 
${CMAKE_CURRENT_SOURCE_DIR}/RootLayerDescription.h 
${CMAKE_CURRENT_SOURCE_DIR}/RootLayeredNetDescription.h 
${CMAKE_CURRENT_SOURCE_DIR}/WeightedLink.h  
${CMAKE_CURRENT_SOURCE_DIR}/SimulationInfoBlock.h 
${CMAKE_CURRENT_SOURCE_DIR}/SimulationInfoBlockVector.h 
${CMAKE_CURRENT_SOURCE_DIR}/LinkDef.h)

add_library( ${MIIND_LIBRARY_PREFIX}clam ${LIB_TYPE} ${clam_SRC}
  ${clam_HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/dict.cxx )

target_link_libraries( ${MIIND_LIBRARY_PREFIX}clam
    ${MIIND_LIBRARY_PREFIX}numtools
    ${MIIND_LIBRARY_PREFIX}util
    ${MIIND_LIBRARY_PREFIX}net
    ${MIIND_LIBRARY_PREFIX}connectionism
    ${MIIND_LIBRARY_PREFIX}structnet
    ${LIB_PREFIX}Core
    ${LIB_PREFIX}RIO
    ${LIB_PREFIX}Cint
    ${LIB_PREFIX}Hist
    ${LIB_PREFIX}Gpad
    ${LIB_PREFIX}Tree
    ${LIB_PREFIX}MathCore
    ${LIB_PREFIX}MathMore
    gsl
    ${CBLAS_PREFIX}cblas )
    

install( FILES ${clam_HEADERS}
    DESTINATION include/miind/ClamLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}clam
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
