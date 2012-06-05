/*
 * EmptyAlgorithm.cpp
 *
 *  Created on: 04.06.2012
 *      Author: david
 */

#include <MPILib/include/Sleep10secAlgorithm.hpp>
#include <boost/thread/thread.hpp>
#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
namespace mpi = boost::mpi;


namespace MPILib {

Sleep10secAlgorithm::Sleep10secAlgorithm(){
	// TODO Auto-generated constructor stub

}

Sleep10secAlgorithm::~Sleep10secAlgorithm() {
	// TODO Auto-generated destructor stub
}

Sleep10secAlgorithm* Sleep10secAlgorithm::Clone() const {
	return new Sleep10secAlgorithm(*this);
}

void Sleep10secAlgorithm::Configure(const SimulationRunParameter& simParam) {

}

void Sleep10secAlgorithm::EvolveNodeState(const std::vector<NodeState>& nodeVector,
		const std::vector<WeightType>& weightVector, Time time) {
	boost::this_thread::sleep( boost::posix_time::seconds(10) );
}

Time Sleep10secAlgorithm::getCurrentTime() const{
	//TODO
	return 1.0;
}

Rate Sleep10secAlgorithm::getCurrentRate() const{
	//TODO
	mpi::communicator world;
	return world.rank()+world.size();

}

} /* namespace MPILib */
