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
///*
#include "CSRAdapter.hpp"
#include "Euler.hpp"

namespace {
	const MPILib::Time TOLERANCE = 1e-6;
}

using namespace TwoDLib;

CSRAdapter::CSRAdapter
(
	Ode2DSystemGroup&                        group,
	const std::vector<TwoDLib::CSRMatrix>&   vecmat,
	MPILib::Rate                             euler_timestep
):
_group(group),
_vec_csr_matrices(vecmat),
_dydt(std::vector<MPILib::Mass>(group.Mass().size())),
_euler_timestep(euler_timestep),
_nr_iterations(this->NumberIterations())
{
}

void CSRAdapter::ClearDerivative()
{
	TwoDLib::ClearDerivative(_dydt);
}

void CSRAdapter::CalculateDerivative
(
	const std::vector<MPILib::Rate>& vecrates
)
{
	TwoDLib::CalculateDerivative
    (
    	_group,
    	_dydt,
    	_vec_csr_matrices,
    	vecrates
    );
}

MPILib::Number CSRAdapter::NumberIterations() const
{
	MPILib::Time tstep = _group.MeshObjects()[0].TimeStep();
    for ( const auto& mesh: _group.MeshObjects() )
        if (std::abs((tstep - mesh.TimeStep())/mesh.TimeStep()) > TOLERANCE){
           std::cerr << "Not all meshes in this group have the same time step. " <<  tstep << " " << mesh.TimeStep() << " " << tstep - mesh.TimeStep()  << std::endl;
           exit(0);
        }
    MPILib::Number  n_steps = static_cast<MPILib::Number>(std::round(tstep/_euler_timestep));

    return n_steps;
}

void CSRAdapter::AddDerivative()
{
	TwoDLib::AddDerivative
		(
			_group.Mass(),
			_dydt,
			_euler_timestep
		);
}
