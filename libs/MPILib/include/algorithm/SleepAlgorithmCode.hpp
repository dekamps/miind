/*
 * EmptyAlgorithm.cpp
 *
 *  Created on: 04.06.2012
 *      Author: david
 */


#ifndef MPILIB_ALGORITHMS_SLEEPALGORITHM_CODE_HPP_
#define MPILIB_ALGORITHMS_SLEEPALGORITHM_CODE_HPP_

#include <boost/thread/thread.hpp>
#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>

#include <MPILib/include/algorithm/SleepAlgorithm.hpp>


namespace mpi = boost::mpi;


namespace MPILib {
namespace algorithm{


template <class WeightValue>
SleepAlgorithm<WeightValue>::SleepAlgorithm(){
	// TODO Auto-generated constructor stub

}

template <class WeightValue>
SleepAlgorithm<WeightValue>::~SleepAlgorithm() {
	// TODO Auto-generated destructor stub
}

template <class WeightValue>
SleepAlgorithm<WeightValue>* SleepAlgorithm<WeightValue>::clone() const {
	return new SleepAlgorithm(*this);
}

template <class WeightValue>
void SleepAlgorithm<WeightValue>::configure(const SimulationRunParameter& simParam) {

//FIXME

}

template <class WeightValue>
void SleepAlgorithm<WeightValue>::evolveNodeState(const std::vector<ActivityType>& nodeVector,
		const std::vector<WeightValue>& weightVector, Time time) {
	time =2;
	unsigned int size = nodeVector.size();
	size = weightVector.size();
//FIXME
	boost::this_thread::sleep( boost::posix_time::seconds(kSleepTime) );

}

template <class WeightValue>
Time SleepAlgorithm<WeightValue>::getCurrentTime() const{
	//TODO
	return 1.0;
}

template <class WeightValue>
Rate SleepAlgorithm<WeightValue>::getCurrentRate() const{
	//TODO
	mpi::communicator world;
	return world.rank()+world.size();

}

template <class WeightValue>
AlgorithmGrid  SleepAlgorithm<WeightValue>::getGrid() const {
	std::vector<double> vector_grid(RATE_STATE_DIMENSION, 1);
	std::vector<double> vector_interpretation(RATE_STATE_DIMENSION, 0);
	return AlgorithmGrid(vector_grid, vector_interpretation);
}



} /* namespace algorithm */
} /* namespace MPILib */

#endif// MPILIB_ALGORITHMS_SLEEP10SECALGORITHM_CODE_HPP_
