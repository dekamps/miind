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
#ifndef _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_LOCALDEFINITIONS_INCLUDE_GUARD
#define _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_LOCALDEFINITIONS_INCLUDE_GUARD

#include <string>

using std::string;

//! SparseImplementationLib

namespace SparseImplementationLib 
{

	const string TAG_ABSTRACTNODE("<AbstractNode>");


	const string STR_LAYEREDIMPLEMENTATION_HEADER
					(
						"<LayeredImplementation>"
					);
	const string STR_SPARSEIMPLEMENTATION_TAG
					(
						"<SparseImplementation>"
					);
	const string STR_LAYEREDSPARSEIMPLEMENTATION_HEADER
					(
						"<LayeredSparseImplementation>"
					);

	const string STR_THRESHOLD_TAG
					(
						"<Threshold>"
					);


	const string STR_LAYERSTARTING_NODES_HEADER
					(
						"<StartLayerNodeIds>"
					);

	const string INVALID_WEIGHT("Weight is NaN");

	const string STR_INVALID_ACTIVATION
					(
						"Node activation is NaN"
					);

	const string STR_THRESHOLD_TAG_EXPECTED
					(
						"Threshold begin or end tag expected"
					);


	const string STR_NODE_PARSING_ERROR
					(
						"Couldn't parse node vector"
					);
	const string STRING_TESTIMPLEMENTATION_NOT_CREATED
					(
						"Failed to create test implementation"
					);

	const string STRING_PREDECESSOR_NOT_FOUND
					(
						"Couldn't locate myself in predecessor list of my successor"
					);

	const string OFFSET_ERROR("Cast to SparseNode failed");

	const string REVERSIBLE_CAST_FAILED("Cast to ReversibleSparseNode failed");

	const string TAG_REVERSIBLE("<ReversibleNode>");

} // end of SparseImplementationlib;

#endif // include guard
