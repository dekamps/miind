// Copyright (c) 2005 - 2010 Marc de Kamps
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
#ifndef _CODE_LIBS_NETLIB_LAYEREDSPARSEIMPLEMENTATION_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_LAYEREDSPARSEIMPLEMENTATION_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <iostream>
#include <memory>
#include "LayerWeightIteratorCode.h"
#include "LayerWeightIteratorThresholdCode.h"
#include "SparseImplementationCode.h"

using std::auto_ptr;
using std::istream;
using NetLib::LayeredArchitecture;
using NetLib::LayeredImplementation;

namespace SparseImplementationLib {

  //! LayeredSparseImplementation
  //! Derives from SparseImplementation, offers support for e.g. Backpropagation

  template <class NodeType>
  class LayeredSparseImplementation : public SparseImplementation<NodeType> {

  public:

    typedef LayerWeightIterator<NodeType>           WeightLayerIterator;
    typedef LayerWeightIteratorThreshold<NodeType>  WeightLayerIteratorThreshold;

    //! Construct from stream (file) 
    LayeredSparseImplementation(istream&);

    //! Construct from LayeredArchitecture
    LayeredSparseImplementation(LayeredArchitecture*);

    //!
    LayeredSparseImplementation(const LayeredSparseImplementation&);

    //!
    LayeredSparseImplementation& operator=(const LayeredSparseImplementation&);

    //!
    virtual ~LayeredSparseImplementation();

    //! layer functions:

    //!
    Number NumberOfLayers		() const;

    //! First Id of a given Layer
    NodeId BeginId (Layer) const;

    //! Last Id of a given Layer
    NodeId EndId   (Layer) const;

    Number NumberOfNodesInLayer	(Layer) const;


	using SparseImplementation<NodeType>::begin;
	using SparseImplementation<NodeType>::end;

    // WeightLayerIteror functions

    //! 
    WeightLayerIterator begin  (Layer, WeightLayerIterator*);

    //!
    WeightLayerIterator end    (Layer, WeightLayerIterator*);

    //!
    WeightLayerIteratorThreshold   begin  (Layer, WeightLayerIteratorThreshold*);

    //!
    WeightLayerIteratorThreshold   end    (Layer, WeightLayerIteratorThreshold*);

    vector<Layer>  ReverseLayerDescription() const;

    // prevent covering of blass function,
    // allow overlading on new type of argument

//    using SparseImplementation<NodeType,NodeValue,WeightValue>::InsertWeight;

    // Streaming functions

    //!.
    virtual bool ToStream(ostream&) const;

    //!
    virtual bool FromStream(istream&);

    //!
    virtual string Tag() const;

  protected:

    // Necessary because of template: disambiguates usef of which template class is used
    using SparseImplementation<NodeType>::_vector_of_nodes;

  private:

    istream&	RemoveHeader(istream&) const;
    istream&	RemoveFooter(istream&) const; 

    NodeId BeginIdNextHighestLayer(Layer) const;
    NodeId EndIdNextHighestLayer  (Layer) const;

    LayeredImplementation	_layered_implementation;

  }; // end of LayeredSparseImplementation

  typedef LayeredSparseImplementation< SparseNode<double,double> > D_LayeredSparseImplementation;
	// inserted by korbo
  typedef LayeredSparseImplementation< SparseNode<float,float> > F_LayeredSparseImplementation;

} // end of SparseImplementationLib


#endif // include guard
