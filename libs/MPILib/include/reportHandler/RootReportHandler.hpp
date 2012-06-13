// Copyright (c) 2005 - 2011 Marc de Kamps
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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net

#ifndef MPILIB_ROOTHANDLER_HPP_
#define MPILIB_ROOTHANDLER_HPP_

#include <string>
#include <vector>
#include <memory>
#include <MPILib/include/reportHandler/AbstractReportHandler.hpp>
#include <MPILib/include/BasicTypes.hpp>
#include <MPILib/include/reportHandler/ValueHandlerHandler.hpp>

// forward declarations 

class TApplication;
class TGraph;
class TNtuple;
class TFile;

namespace MPILib {
//! This handler writes states and firing rates as TGraph objects in a root file.
//! (see http://root.cern.ch). It is also able to show run time graphs of selected nodes.
//!
//! ROOT is a visualisation and data management tool with a good interface to numerical
//! methods. The RootReportHandler is reliable when no run time graphs are shown and is a very
//! efficient way to store the simulation data, as they are compressed when written to file.
//! The behaviour and resource consumption of the run time graphs need closer investigation
//! but for debugging purposes they are tremendously useful. A disadvantage for high throughput is that
//! memory use increases over time. Where this is an issue, use RootHighThroughputHandler.
class RootReportHandler: public AbstractReportHandler {
public:

	RootReportHandler(const std::string&, bool b_force_state_write = false);

	RootReportHandler(const RootReportHandler&);

	//! virtual destructor
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
	 * @param The NodeId of the Node
	 */
	virtual void initializeHandler(const NodeId&);

	/**
	 * A MPINode will request to be dissociated from the handler at the end of simulation.
	 * @param The NodeId of the Node
	 */
	virtual void detachHandler(const NodeId&);

private:

	/**
	 * Writes the info tuple to the file
	 * @param The NodeId
	 */
	void writeInfoTuple(const NodeId&);

	/**
	 * removes the node from the handler
	 * @param The NodeId of the node removed
	 */
	void removeFromNodeList(NodeId);

	/**
	 * Finalize the report Handler
	 */
	void finalize();

	/**
	 * TODO what does this
	 * @param
	 * @return
	 */
	std::unique_ptr<TGraph> convertAlgorithmGridToGraph(const Report&) const;

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
	 * Pointer to the file. @todo change this to smart pointer
	 */
	static TFile* _pFile;

	/**
	 * Pointer to the tuple. @todo change this to smart pointer
	 */
	static TNtuple* _pTuple;

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
	bool _isStateWriteMandatory {false};

	/**
	 * Number of reports generated so far
	 */
	int _nrReports {0};

};

} // end of MPILib

#endif // MPILIB_ROOTHANDLER_HPP_ include guard
