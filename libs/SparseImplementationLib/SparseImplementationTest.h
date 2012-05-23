// Copyright (c) 2005 - 2009 Marc de Kamps
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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_SPARSEIMPLEMENTATIONTEST_INCLUDE_GUARD
#define _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_SPARSEIMPLEMENTATIONTEST_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <iostream>
#include "../NetLib/NetLib.h"
#include "../NetLib/NetLibTest.h"
#include "../UtilLib/UtilLib.h"
#include "ReversibleLayeredSparseImplementationCode.h"

using std::ostream;
using NetLib::NetLibTest;
using UtilLib::LogStream;


namespace SparseImplementationLib
{

	class SparseImplementationTest : public LogStream
	{
	public:

		SparseImplementationTest();         // no log file written

		SparseImplementationTest
		(
			boost::shared_ptr<ostream>				// stream for the log file
		); 

		virtual ~SparseImplementationTest(){}

		bool Execute();

		// a small SparseImplementation for testing purposes
		template <class NodeType>
			LayeredSparseImplementation<NodeType>
				CreateTestLayeredSparseImplementation() const;
		template <class ReversibleNode>
			ReversibleLayeredSparseImplementation<ReversibleNode>
				CreateTestReversibleLayeredSparseImplementation() const;

		LayeredSparseImplementation<D_ReversibleSparseNode>
			CreateXORImplementation() const;

		LayeredArchitecture
			CreateXORArchitecture  () const;

	private:


		bool SparseNodeCreationTest				() const;
		bool SparseNodeCopyTest					() const;
		bool SparseImplementationTesting		() const;
		bool SparseImplementationCopyTest		() const;
		bool SparseImplementationReversalTest	() const;
		bool PredecessorIteratorTest			() const;
		bool LayerOrderTest						() const;
		bool LayerIteratorTest					() const;
		bool LayerIteratorThresholdTest			() const;
		bool ReverseIteratorTest				() const;
		bool ReverseIteratorThresholdTest		() const;
		bool XORImplementationTest				() const;
		bool TestSparseNodeStreaming			() const;
		bool TestImplementationStreaming		() const;
		bool DynamicInsertionTest				() const;
		bool SparseImplementationAllocatorTest	() const;
		bool NodeVectorTest						() const;
		bool ImplementationCreationTest			() const;
		bool NodeLinkCollectionStreamTest		() const;
		bool LayeredArchitectureTest			() const;
		bool NavigationTest						() const;
	
		bool TestReverseInnerProductLayer1
			(
				D_LayeredReversibleSparseImplementation& 
			) const;
		bool TestReverseInnerProductLayer2
			( 
				D_LayeredReversibleSparseImplementation&
			) const;
	
		bool TestReverseThresholdInnerProductLayer1
			(
				D_LayeredReversibleSparseImplementation& 
			) const;
	        bool TestReverseThresholdInnerProductLayer2
			( 
				D_LayeredReversibleSparseImplementation&
			) const;

		const string _path_name;

	}; // end of SparseNodeTest

 } // end of SparseImplementationLib

#endif // include guard
