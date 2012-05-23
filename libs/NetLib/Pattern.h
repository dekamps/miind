// Copyright (c) 2005 - 2008 Marc de Kamps
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
#ifndef _CODE_LIBS_NETLIB_PATTERN_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_PATTERN_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <cassert>
#include <cstddef>
#include <string>
#include <vector>
#include "../NumtoolsLib/NumtoolsLib.h"
#include "../UtilLib/UtilLib.h"
#include "PatternParsingException.h"

using std::istream;
using std::ostream;
using std::vector;
using std::iterator;
using UtilLib::Index;
using UtilLib::Streamable;
using NumtoolsLib::RandomGenerator;
using NumtoolsLib::UniformDistribution;

// Created:				14-06-1999
// Author:				Marc de Kamps


namespace NetLib {

  //! Pattern

  template <class PatternValue>
    class Pattern : public Streamable {
    public:

    //! ctor
    Pattern();

    //! ctor, predefined size
    Pattern( Number );

    //! copy ctor
    Pattern( const Pattern& );

    //!  dtor
    virtual ~Pattern();

    //! copy operator
    Pattern& operator=( const Pattern& );
	
		// element access:

    //!
    PatternValue&	operator[]( Index );

    //!
    const PatternValue&	operator[]( Index ) const;

    //! Pattern length
    Number Size() const;

    //! Set all elements to 0
    void Clear();

    //! Create random pattern
    void RandomizeBinary(RandomGenerator&);

    //!
    void ClipMax( PatternValue);

    //!
    void ClipMin( PatternValue );

    //! Add two patterns
    Pattern& operator+=( const Pattern&);

		// streaming functions:

    //! XML Tag for serialization
    virtual string Tag () const;

    //! Output streaming function
    virtual bool ToStream  (ostream&) const;

    //! Build up from stream
    virtual bool FromStream(istream&);


    template <class P> friend istream& operator>>(istream&,       Pattern<P>& );
    template <class P> friend ostream& operator<<(ostream&, const Pattern<P>& );

    typedef typename vector<PatternValue>::iterator pattern_iterator;

    //! pattern iterator: begin
    pattern_iterator begin();

    //! pattern iterator: end
    pattern_iterator end  ();

    protected:

    vector<PatternValue> _vector_of_pattern;

    private:

  };

	//! serialization: output
	template <class PatternValue>	ostream& operator<<
	(
		ostream&, 
		const Pattern<PatternValue>&
	);

	//! serialization: input
	template <class PatternValue> istream& operator>>
	( 
		istream&, 
		Pattern<PatternValue>&	
	);

	//! add patterns
	template <class PatternValue> Pattern<PatternValue> operator+
	(
		const Pattern<PatternValue>&,
		const Pattern<PatternValue>&
	);

  enum OnOrOff { Off = 0,  On  = 1 };
  enum PosOrNeg{ Min = -1, Max = 1 };

  typedef Pattern<double>		D_Pattern;
  typedef Pattern<PosOrNeg>	SignedBinaryPattern;

} // end of NetLib


#endif // include guard
