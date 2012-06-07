/*
 * WilsonCowanAlgorithm.hpp
 *
 *  Created on: 07.06.2012
 *      Author: david
 */

#ifndef MPILIB_WILSONCOWANALGORITHM_HPP_
#define MPILIB_WILSONCOWANALGORITHM_HPP_

#include "../../NumtoolsLib/NumtoolsLib.h"
#include "WilsonCowanParameter.hpp"
#include <MPILib/include/AlgorithmInterface.hpp>

namespace MPILib {

class WilsonCowanAlgorithm: public AlgorithmInterface<double> {
public:
	WilsonCowanAlgorithm();

	WilsonCowanAlgorithm(const WilsonCowanParameter&);

	virtual ~WilsonCowanAlgorithm();

	/**
	 * Cloning operation, to provide each DynamicNode with its own
	 * Algorithm instance. Clients use the naked pointer at their own risk.
	 */
	virtual WilsonCowanAlgorithm* Clone() const;

	/**
	 * Configure the Algorithm
	 * @param simParam
	 */
	virtual void Configure(const SimulationRunParameter& simParam);

	/**
	 * Evolve the node state
	 * @param nodeVector Vector of the node States
	 * @param weightVector Vector of the weights of the nodes
	 * @param time Time point of the algorithm
	 */
	virtual void EvolveNodeState(const std::vector<Rate>& nodeVector,
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

private:

	double innerProduct(const std::vector<Rate>& nodeVector,
			const std::vector<double>& weightVector);

	vector<double> InitialState() const;

	WilsonCowanParameter _parameter;

	NumtoolsLib::DVIntegrator<WilsonCowanParameter> _integrator;

};

} /* namespace MPILib */
#endif /* MPILIB_WILSONCOWANALGORITHM_HPP_ */
