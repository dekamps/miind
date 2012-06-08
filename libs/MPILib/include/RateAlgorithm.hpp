/*
 * WilsonCowanAlgorithm.hpp
 *
 *  Created on: 07.06.2012
 *      Author: david
 */

#ifndef MPILIB_RATEALGORITHM_HPP_
#define MPILIB_RATEALGORITHM_HPP_

#include <NumtoolsLib/NumtoolsLib.h>
#include <DynamicLib/WilsonCowanParameter.h>

#include <MPILib/include/AlgorithmInterface.hpp>

namespace MPILib {

class RateAlgorithm: public AlgorithmInterface<double> {
public:

	RateAlgorithm(Rate* rate);

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
	virtual void configure(const DynamicLib::SimulationRunParameter& simParam);

	/**
	 * Evolve the node state
	 * @param nodeVector Vector of the node States
	 * @param weightVector Vector of the weights of the nodes
	 * @param time Time point of the algorithm
	 */
	virtual void evolveNodeState(const std::vector<Rate>& nodeVector,
			const std::vector<double>& weightVector, Time time);

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

	virtual DynamicLib::AlgorithmGrid getGrid() const;

private:

	Time _time_current;
	Rate _rate;
	Rate* _p_rate;

};

} /* namespace MPILib */
#endif /* MPILIB_RATEALGORITHM_HPP_ */
