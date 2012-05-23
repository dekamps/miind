project( ${MIIND_LIBRARY_PREFIX}clam )

if( WIN32 )
  find_package( WINGSL REQUIRED )
  if( NOT GSL_FOUND )
    message( FATAL_ERROR "WinGsl not found (module Populist)." )
  endif( NOT GSL_FOUND )
  set( LIB_TYPE STATIC )
else( WIN32 )
  include( RequireGSL )
  if ( APPLE )             #This is not necessary any more under leopard
    set( LIB_TYPE STATIC ) #but it makes no differnece
  else( APPLE )
    set( LIB_TYPE SHARED )
  endif( APPLE )
endif( WIN32 )

include_directories( ${GSL_INCLUDE_DIRS} )
link_directories( ${GSL_LIBRARY_DIRS} )

find_package( Root )
if( NOT ROOT_FOUND )
    message( FATAL_ERROR "Root Library not found. Is essential for ClamLib." )
endif( NOT ROOT_FOUND )

include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )

set( agent_SRC
AbstractNetworkTraderAlgorithm.cpp
ChartistNetworkTraderAlgorithm.cpp
ChartistTraderAlgorithm.cpp
ChiarellasModelSimulator.cpp
CMRootReportHandler.cpp
FundamentalistNetworkTraderAlgorithm.cpp
FundamentalistTraderAlgorithm.cpp
MarketMakerAlgorithm.cpp
NetworkedMarketMakerAlgorithm.cpp
NetworkedModelSimulator.cpp
NetworkedRootReportHandler.cpp
NetworkTopology.cpp
RandSimulator.cpp
WalrasianAuctioneerAlgorithm.cpp
WalrasianAuctioneer.cpp
WalrasianTraderAlgorithm.cpp

)

set( agent_HEADERS
AbstractNetworkTraderAlgorithm.h
ChartistNetworkTraderAlgorithm.h
ChartistTraderAlgorithm.h
ChiarellasModelSimulator.h
CMRootReportHandler.h
FundamentalistNetworkTraderAlgorithm.h
FundamentalistTraderAlgorithm.h
MarketMakerAlgorithm.h
NetworkedMarketMakerAlgorithm.h
NetworkedModelSimulator.h
NetworkedRootReportHandler.h
NetworkTopology.h
RandSimulator.h
Scheduler.h
WalrasianAuctioneerAlgorithm.h
WalrasianAuctioneer.h
WalrasianTraderAlgorithm.h
)


add_library( ${MIIND_LIBRARY_PREFIX}agent ${LIB_TYPE} ${agent_SRC}
  ${agent_HEADERS} )

target_link_libraries( ${MIIND_LIBRARY_PREFIX}agent
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
    gsl
    ${CBLAS_PREFIX}cblas )
    

install( FILES ${agent_HEADERS}
    DESTINATION include/miind/AgentLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}agent
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
