/*
 * SleepAlgorithm.hpp
 *
 *  Created on: 20.06.2012
 *      Author: david
 */

#ifndef SLEEPALGORITHM_HPP_
#define SLEEPALGORITHM_HPP_
#include <MPILib/include/algorithm/AlgorithmGrid.hpp>
#include <boost/thread/thread.hpp>
#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>

using namespace MPILib;

template<class WeightValue>
class SleepAlgorithm: public algorithm::AlgorithmInterface<WeightValue> {
public:

	SleepAlgorithm() {

	}
	~SleepAlgorithm() {
	}

	SleepAlgorithm<WeightValue>* clone() const {
		return new SleepAlgorithm(*this);
	}

	void configure(const SimulationRunParameter&) {

	}

	void evolveNodeState(const std::vector<ActivityType>&,
			const std::vector<WeightValue>&, Time) {

		boost::this_thread::sleep(boost::posix_time::seconds(kSleepTime));

	}

	Time getCurrentTime() const {
		return 1.0;
	}

	Rate getCurrentRate() const {
		mpi::communicator world;
		return world.rank() + world.size();

	}

	algorithm::AlgorithmGrid getGrid() const {
		std::vector<double> vector_grid(RATE_STATE_DIMENSION, 1);
		std::vector<double> vector_interpretation(RATE_STATE_DIMENSION, 0);
		return algorithm::AlgorithmGrid(vector_grid, vector_interpretation);
	}

private:

	double kSleepTime = 1;
};

#endif /* SLEEPALGORITHM_HPP_ */
