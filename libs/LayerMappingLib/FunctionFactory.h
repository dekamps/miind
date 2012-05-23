// Copyright (c) 2005 - 2007 Marc de Kamps, Johannes Drever, Melanie Dietz
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

#ifndef LAYERMAPPINGLIB_FUNCTIONFACTORY_H
#define LAYERMAPPINGLIB_FUNCTIONFACTORY_H

#include <sstream>

using namespace std;

#include "AbstractFunction.h"
#include "Exception.h"

//Register your function here
#include "Functions/MinCode.h"
#include "Functions/MaxCode.h"
#include "Functions/MeanCode.h"
#include "Functions/StandardDeviationCode.h"
#include "Functions/SumCode.h"
#include "Functions/ProductCode.h"
#include "Functions/ConvolutionCode.h"
#include "Functions/CompositeFeatureCode.h"
#include "Functions/PerceptronCode.h"
#include "Functions/ArgMaxCode.h"
#include "Functions/IdentityCode.h"
#include "Functions/FFTRCode.h"
#include "Functions/FFTCCode.h"
#include "Functions/CombineCode.h"
#include "Functions/ScaleCode.h"
#include "Functions/ProductComplexRealCode.h"
#include "Functions/ProductComplexImagCode.h"
#include "Functions/SimpleCellResponseCode.h"

namespace LayerMappingLib
{
	//Register your function here
	static const string MAX = "Max";
	static const string MIN = "Min";
	static const string MEAN = "Mean";
	static const string STANDARD_DEVIATION = "StandardDeviation";
	static const string SUM = "Sum";
	static const string PRODUCT = "Product";
	static const string COMPOSITE_FEATURE = "CompositeFeature";
	static const string CONVOLUTION = "Convolution";
	static const string PERCEPTRON = "Perceptron";
	static const string ARGMAX = "ArgMax";
	static const string IDENTITY = "Identity";
	static const string COMBINE = "Combine";
	static const string SCALE = "Scale";
	static const string SIMPLE_CELL_RESPONSE = "SimpleCellResponse";
	#ifdef HAVE_FFTW
	static const string FFTR_ = "FFTR";
	static const string FFTC_ = "FFTC";
	static const string FFT_FORWARD = "forward";
	static const string FFT_BACKWARD = "backward";
	static const string FFT_TYPE_REAL = "real";
	static const string FFT_TYPE_IMAG = "imag";
	#endif //HAVE_FFTW
	static const string PRODUCT_COMPLEX_REAL = "ProductComplexReal";
	static const string PRODUCT_COMPLEX_IMAG = "ProductComplexImag";

	template<class Function>
	struct FunctionFactory
	{
		typedef typename Function::vector_list vector_list;

		static Function* get_function( iostream& s );
		static Function* empty_function();
	};
}

#endif //LAYERMAPPINGLIB_FUNCTIONFACTORY_H
