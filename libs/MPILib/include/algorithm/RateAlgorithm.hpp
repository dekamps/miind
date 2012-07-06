/*
 * WilsonCowanAlgorithm.hpp
 *
 *  Created on: 07.06.2012
 *      Author: david
 */

#ifndef MPILIB_ALGORITHMS_RATEALGORITHM_HPP_
#define MPILIB_ALGORITHMS_RATEALGORITHM_HPP_


#include <MPILib/include/algorithm/AlgorithmInterface.hpp>

namespace MPILib {
namespace algorithm{

template<class Weight>
class RateAlgorithm: public AlgorithmInterface<Weight> {
public:

	/**
	 * Constructor
	 * @param rate the rate of the algorithm
	 */
	RateAlgorithm(Rate rate);

	virtual ~RateAlgorithm();

	/**
	 * Cloning operation, to provide each DynamicNode with its own
	 * Algorithm instance. Clients use the naked pointer at their own risk.
	 */
	virtual RateAlgorithm* clone() const;

	/**
	 * Configure the Algorithm
	 * @param simParam
	 */
	virtual void configure(const SimulationRunParameter& simParam);

	/**
	 * Evolve the node state
	 * @param nodeVector Vector of the node States
	 * @param weightVector Vector of the weights of the nodes
	 * @param time Time point of the algorithm
	 */
	virtual void evolveNodeState(const std::vector<Rate>& nodeVector,
			const std::vector<Weight>& weightVector, Time time);

	/**
	 * The current timepoint
	 * @return The current time point
	 */
	virtual Time getCurrentTime() const;

	/**
	 * The calculated rate of the node
	 * @return The rate of the node
	 */
	virtual Rate getCurrentRate() const;

	virtual AlgorithmGrid getGrid() const;

private:

	Time _time_current;
	Rate _rate;

};

} /* namespace algorithm */
} /* namespace MPILib */
#endif /* MPILIB_ALGORITHMS_RATEALGORITHM_HPP_ */
