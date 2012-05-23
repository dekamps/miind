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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_NETLIB_SIGMOID_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_SIGMOID_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <iostream>
#include "AbstractSquashingFunction.h"
#include "SigmoidParameter.h"

using std::istream;
using std::ostream;


namespace NetLib {

  //! Sigmoid

  class Sigmoid : public AbstractSquashingFunction  {
    public:

    //! Default constructor, setting default parameters
    Sigmoid();

    //!
    Sigmoid(istream&);

    //! Construct a Sigmoid, using the parameters
    Sigmoid(const SigmoidParameter& par);

    //! destructor
    virtual ~Sigmoid();

    //  1/( 1 + exp -x ) and variants 
    //! 1/( 1 + exp -x ) and variants 
    virtual double operator()( double ) const;

    //  analytic expression for derivative (important for Backpropagation)
    //! analytic expression for derivative (important for Backpropagation)
    virtual double Derivative( double ) const;

    //!
    virtual double MinimumActivity() const;

    //!
    virtual double MaximumActivity() const;

    //!
    virtual AbstractSquashingFunction* Clone() const ;

    //!
    virtual SigmoidParameter& 
      GetSquashingParameter();

    // streaming functions

    //!
    virtual string Tag() const;

    //!
    virtual bool ToStream  (ostream&) const;

    //!
    virtual bool FromStream(istream&);

  private:

    double Inverse( double ) const;
			
    SigmoidParameter _parameter_squash;

  };

  ostream& operator<<(ostream&, const Sigmoid&);	
  istream& operator>>(istream&,  Sigmoid&);


} // end of ImplementationLib

#endif // include guard
