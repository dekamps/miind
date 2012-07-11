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

#ifndef MPILIB_REPORT_HANDLER_ROOTHIGHTHROUGHPUTHANDLER_HPP_
#define MPILIB_REPORT_HANDLER_ROOTHIGHTHROUGHPUTHANDLER_HPP_

#include <string>
#include <vector>
#include <map>
#include <MPILib/include/report/handler/AbstractReportHandler.hpp>
#include <MPILib/include/TypeDefinitions.hpp>

class TTree;
class TFile;
class TBranch;
template<class Type> class TVectorT;

namespace MPILib {
namespace report {
namespace handler {
/**
 * This is a handler which organizes data in time slices.
 * Its memory use is constant over a simulation, unlike that
 * of RootReportHandler.
 */
class RootHighThroughputHandler: public AbstractReportHandler {
public:

	/**
	 * Standard constructor for client code
	 * @param fileName root file name without extension!
	 * @param generateGraphs backward compatibility option for an older ROOT layout
	 */
	RootHighThroughputHandler(const std::string& fileName, bool generateGraphs = false);

	/**
	 * Copy constructor
	 * @param rhs another RootHighThroughputHandler
	 */
	RootHighThroughputHandler(const RootHighThroughputHandler& rhs);

	virtual ~RootHighThroughputHandler();

	/**
	 * Writes the Report to the file.
	 * @param report The report written into the file
	 */
	virtual void writeReport(const Report&);

	/**
	 * Cloning operation
	 * @return A clone of the ReportHandler
	 */
	virtual RootHighThroughputHandler* clone() const;

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

private:

	/**
	 * Generate Noded graphs from all nodes stored in this file
	 * @param fileName The name of the file
	 */
	void generateNodeGraphs(const char* fileName);

	/**
	 * Collects the number of node, slices and the time from the TTree
	 * @param vecTime Vector where the Time Points are stored
	 * @param nrNodes contains at the end the number of nodes
	 * @param nrSlices contains at the end the number of slices
	 */
	void collectGraphInformation(std::vector<double>& vecTime, Number& nrNodes,
			Number& nrSlices);

	/**
	 * Stores the Graphs in the TFile
	 * @param vecTime A vector of the time points
	 * @param nrNodes The number of nodes stored in this file
	 * @param nrSlices The number of slices stored in this file
	 */
	void storeRateGraphs(const std::vector<double>& vecTime, Number nrNodes, Number nrSlices);

	/**
	 * Pointer to the TTree
	 */
	static TTree* _pTree;
	/**
	 * Pointer to the TFile
	 */
	static TFile* _pFile;
	/**
	 * TVector of the stored rates
	 */
	static TVectorT<double>* _pArray;
	/**
	 * True if Graphs of the rates should be generated
	 */
	static bool _generateNodeGraphs;
	/**
	 * True if the initialization is finished
	 */
	static bool _isRecording;
	/**
	 * Storage for the node rates, to allow asynchronous calls to the write method
	 */
	static std::map<int, double> _mData;

};

} // end namespace of handler
} // end namespace of report
} // end namespace of MPILib
#endif // include guard MPILIB_REPORT_HANDLER_ROOTHIGHTHROUGHPUTHANDLER_HPP_
