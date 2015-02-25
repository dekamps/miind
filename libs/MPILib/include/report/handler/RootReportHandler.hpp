// Copyright (c) 2005 - 2012 Marc de Kamps
//						2012 David-Matthias Sichau
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation
//      and/or other materials provided with the distribution.
//    * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software
//      without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#ifndef MPILIB_REPORT_HANDLER_ROOTHANDLER_HPP_
#define MPILIB_REPORT_HANDLER_ROOTHANDLER_HPP_

#include <string>
#include <vector>
#include <memory>
#include <MPILib/include/report/handler/AbstractReportHandler.hpp>
#include <MPILib/include/TypeDefinitions.hpp>
#include "CanvasParameter.hpp"
#include "RootCanvasNoMPI.hpp"
#include "ValueHandlerHandler.hpp"

// forward declarations 

class TGraph;
class TFile;


namespace MPILib {
namespace report {
namespace handler {

/**
 * @brief This handler writes states and firing rates as TGraph objects in a root file.
 * (see http://root.cern.ch). It is also able to show run time graphs of selected nodes.
 *
 * ROOT is a visualisation and data management tool with a good interface to numerical
 * methods. The RootReportHandler is reliable when no run time graphs are shown and is a very
 * efficient way to store the simulation data, as they are compressed when written to file.
 * The behaviour and resource consumption of the run time graphs need closer investigation
 * but for debugging purposes they are tremendously useful. A disadvantage for high throughput is that
 * memory use increases over time. Where this is an issue, use RootHighThroughputHandler.
 */
class RootReportHandler: public AbstractReportHandler {
public:

	/**
	 * Constructor
	 * @param file_name The name of the root file
	 * @param b_force_state_write Set true if the state should be written to the root file
	 * @param bOnScreen Set true if a running simulation must be shown on a Canvas. Ineffective if compiled with MPI_ENABELED
	 */
	RootReportHandler(const std::string& file_name, bool b_force_state_write = false, bool bOnCanvas = false, const CanvasParameter& = DEFAULT_CANVAS);

	/**
	 * Copy constructor
	 * @param rhs another RootReportHandler
	 */
	RootReportHandler(const RootReportHandler& rhs);

	/**
	 * virtual destructor
	 */
	virtual ~RootReportHandler();

	/**
	 * Writes the Report to the file.
	 * @param report The report written into the file
	 */
	virtual void writeReport(const Report&);

	/**
	 * Cloning operation
	 * @return A clone of the algorithm
	 */
	virtual RootReportHandler* clone() const;

	/**
	 * During Configuration a MPINode will associate itself with the handler.
	 * @param nodeId The NodeId of the Node
	 */
	virtual void initializeHandler(const NodeId& nodeId);

	/**
	 * A MPINode will request to be dissociated from the handler at the end of simulation.
	 * @param nodeId The NodeId of the Node
	 */
	virtual void detachHandler(const NodeId& nodeId);

	/**
	 * To show a running simulation a node must be adde to a canvas. The state and rate
	 * of the node will be rendered as the simulation progresses.
	 */
	void addNodeToCanvas(NodeId);

	/**
	 *  get the parameter that was used to configure the canvas;
	 */
	CanvasParameter getCanvasParameter() const;

private:


	/**
	 * removes the node from the handler
	 * @param nodeId The NodeId of the node removed
	 */
	void removeFromNodeList(NodeId nodeId);

	/**
	 * Finalize the report Handler
	 */
	void finalize();

	/**
	 * convert the state of the algorithm to a graph
	 * @param report The report
	 * @return A graph
	 */
	std::unique_ptr<TGraph> convertAlgorithmGridToGraph(const Report& report) const;

	/**
	 * Check if the Handler is connected to an algorithm
	 * @return True if the Handler is connected to an algorithm
	 */
	bool isConnectedToAlgorithm() const;

	/**
	 * Does the current State need to be written to the file
	 * @return True if the State need to be written to the file
	 */
	bool isStateWriteMandatory() const;

	/**
	 * Pointer to the file.
	 */
	static TFile* _pFile;

	/**
	 * The Value Handler
	 */
	static ValueHandlerHandler _valueHandler;

	/**
	 * Vector of the connected Nodes
	 */
	static std::vector<NodeId> _nodes;


	/**
	 * Pointer to the current rate graph
	 */
	std::unique_ptr<TGraph> _spCurrentRateGraph;
	/**
	 * Pointer to the current state graph
	 */
	std::unique_ptr<TGraph> _spCurrentStateGraph;

	/**
	 * True if the state should be written to the file
	 */
	bool _isStateWriteMandatory = false;

	/**
	 * Number of reports generated so far
	 */
	int _nrReports = 0;

	bool _bOnCanvas = false;

	RootCanvas _canvas;
};

}// end namespace of handler
}// end namespace of report
}// end namespace of MPILib

#endif // MPILIB_REPORT_HANDLER_ROOTHANDLER_HPP_ include guard
