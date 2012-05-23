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
#include "../ClamLib/ClamLib.h"
#include "../PopulistLib/PopulistLib.h"
#include "../DynamicLib/DynamicLib.h"

using ClamLib::SemiSigmoid;

namespace DynamicLib {

	template <> boost::shared_ptr< AbstractAlgorithm<PopulistLib::PopulationConnection> > 
		AlgorithmBuilder<PopulistLib::PopulationConnection>::Build(istream& s) const
        {
			typedef boost::shared_ptr<AbstractAlgorithm<PopulationConnection> > palg;
			string alg_name = this->ParseHeader(s);

	
			if (alg_name == "invalid"){
				palg p_alg = palg(new RateAlgorithm<PopulationConnection>(0.0));
				p_alg->SetValidity(false);
				p_alg->SetOffendingString(_offending_string);
				return p_alg;
			}

			if (alg_name == "PopulationAlgorithm"){
			  palg p_alg(new PopulationAlgorithm(s));
			  p_alg->SetValidity(true);
			  return p_alg;
			}

			if (alg_name == "RateAlgorithm"){
			  palg p_alg(new RateAlgorithm<PopulationConnection>(s));
			  p_alg->SetValidity(true);
			  return p_alg;
			}

			if (alg_name == "OrnsteinUhlenbeckAlgorithm"){
			  palg p_alg(new OrnsteinUhlenbeckAlgorithm(s));
			  p_alg->SetValidity(true);
			  return p_alg;
			}

			if (alg_name == "DelayAlgorithm"){
				palg p_alg(new DelayAlgorithm<PopulationConnection>(s));
				p_alg->SetValidity(true);
				return p_alg;
			}

			throw DynamicLibException("Unknown algorithm in builder");
        }
}
