/*
 * EmptyAlgorithm.cpp
 *
 *  Created on: 04.06.2012
 *      Author: david
 */


#ifndef MPILIB_ALGORITHMS_SLEEP10SECALGORITHM_CODE_HPP_
#define MPILIB_ALGORITHMS_SLEEP10SECALGORITHM_CODE_HPP_

#include <boost/thread/thread.hpp>
#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>

#include <MPILib/include/algorithm/Sleep10secAlgorithm.hpp>


namespace mpi = boost::mpi;


namespace MPILib {
namespace algorithm{


template <class WeightValue>
Sleep10secAlgorithm<WeightValue>::Sleep10secAlgorithm(){
	// TODO Auto-generated constructor stub

}

template <class WeightValue>
Sleep10secAlgorithm<WeightValue>::~Sleep10secAlgorithm() {
	// TODO Auto-generated destructor stub
}

template <class WeightValue>
Sleep10secAlgorithm<WeightValue>* Sleep10secAlgorithm<WeightValue>::clone() const {
	return new Sleep10secAlgorithm(*this);
}

template <class WeightValue>
void Sleep10secAlgorithm<WeightValue>::configure(const DynamicLib::SimulationRunParameter& simParam) {

//FIXME

}

template <class WeightValue>
void Sleep10secAlgorithm<WeightValue>::evolveNodeState(const std::vector<ActivityType>& nodeVector,
		const std::vector<WeightValue>& weightVector, Time time) {
	time =2;
	unsigned int size = nodeVector.size();
	size = weightVector.size();
//FIXME
	boost::this_thread::sleep( boost::posix_time::seconds(10) );

}

template <class WeightValue>
Time Sleep10secAlgorithm<WeightValue>::getCurrentTime() const{
	//TODO
	return 1.0;
}

template <class WeightValue>
Rate Sleep10secAlgorithm<WeightValue>::getCurrentRate() const{
	//TODO
	mpi::communicator world;
	return world.rank()+world.size();

}

template <class WeightValue>
DynamicLib::AlgorithmGrid  Sleep10secAlgorithm<WeightValue>::getGrid() const {
	std::vector<double> vector_grid(DynamicLib::RATE_STATE_DIMENSION, 1);
	std::vector<double> vector_interpretation(DynamicLib::RATE_STATE_DIMENSION, 0);
	return DynamicLib::AlgorithmGrid(vector_grid, vector_interpretation);
}

} /* namespace algorithm */
} /* namespace MPILib */

#endif// MPILIB_ALGORITHMS_SLEEP10SECALGORITHM_CODE_HPP_
