// Copyright (c) 2005 - 2015 Marc de Kamps
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

#ifndef _CODE_CUDA2DLIB_CSRAdapter_INCLUDE_GUARD
#define _CODE_CUDA2DLIB_CSRAdapter_INCLUDE_GUARD

#include <cassert>
#include <cuda_runtime.h>
#include "../TwoDLib/TwoDLib.hpp"
#include "CudaOde2DSystemAdapter.cuh"


namespace CudaTwoDLib {

	  /**
	   * \brief  Responsible for maintaining the mirror of Master equation calls  on a GPGPU device
	   *
	   * Executes the Master equation solver on the GPU device
	   */

        //! floating point type for cuda system
	typedef float fptype;
	typedef unsigned int inttype;

	class CSRAdapter {
	public:

							CSRAdapter(CudaOde2DSystemAdapter& adapter, const std::vector<TwoDLib::CSRMatrix>& matrixvector, fptype euler_timestep );

							CSRAdapter(CudaOde2DSystemAdapter& group, const std::vector<TwoDLib::CSRMatrix>& vecmat,
							 inttype nr_connections, fptype euler_timestep,
							 const std::vector<inttype>& vecmat_indexes, const std::vector<inttype>& grid_transforms);

              ~CSRAdapter();

              void InspectMass(inttype);

              void ClearDerivative();

              void CalculateDerivative(const std::vector<fptype>&);

							void CalculateMeshGridDerivative(const std::vector<inttype>&, const std::vector<fptype>&, const std::vector<fptype>&, const std::vector<fptype>&, const std::vector<int>&, const std::vector<int>&);

							void CalculateGridDerivative(const std::vector<inttype>&, const std::vector<fptype>&, const std::vector<fptype>&, const std::vector<fptype>&, const std::vector<int>&, const std::vector<int>&);

							void SingleTransformStep();

              void AddDerivative();

							void AddDerivativeFull();

              inttype NrIterations() const { return _nr_iterations; }

        private:

              inttype NumberIterations(const CudaOde2DSystemAdapter&, fptype) const;

              void FillMatrixMaps(const std::vector<TwoDLib::CSRMatrix>&);
              void DeleteMatrixMaps();
              void FillDerivative();
              void DeleteDerivative();
              void CreateStreams();
              void DeleteStreams();

	          	std::vector<inttype> Offsets(const std::vector<TwoDLib::CSRMatrix>&) const;
	          	std::vector<inttype> NrRows(const std::vector<TwoDLib::CSRMatrix>&) const;

	      			CudaOde2DSystemAdapter& _group;
              fptype                  _euler_timestep;
              inttype                 _nr_iterations;
              inttype                 _nr_m;
							inttype									_nr_streams;

							std::vector<inttype>		_grid_transforms;
							std::vector<inttype> 		_vecmats;

              std::vector<inttype>   _nval;
              std::vector<fptype*>   _val;
              std::vector<inttype>   _nia;
              std::vector<inttype*>  _ia;
              std::vector<inttype>   _nja;
              std::vector<inttype*>  _ja;

              std::vector<inttype> _offsets;
              std::vector<inttype> _nr_rows;

              fptype* _dydt;

              int _blockSize;
              int _numBlocks;

              cudaStream_t* _streams;
	};
}
#endif // include guard
