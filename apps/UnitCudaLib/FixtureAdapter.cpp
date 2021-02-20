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

#include <boost/test/unit_test.hpp>
#include <CudaTwoDLib.hpp>

#include "FixtureAdapter.hpp"

std::vector<TwoDLib::Mesh>  FixtureAdapter::Mesh() const
{
       pugi::xml_document doc;
       std::vector<TwoDLib::Mesh> vec_vec_mesh;
       std::vector< std::vector<TwoDLib::Redistribution> > vec_vec_rev;
       std::vector< std::vector<TwoDLib::Redistribution> > vec_vec_res;
       pugi::xml_parse_result result0 = doc.load_file("condee2a5ff4-0087-4d69-bae3-c0a223d03693.model");
       pugi::xml_node  root0 = doc.first_child();
       TwoDLib::Mesh mesh0 = TwoDLib::RetrieveMeshFromXML(root0);
       std::vector<TwoDLib::Mesh> vec_mesh{ mesh0 };
       return vec_mesh;
}

std::vector< std::vector<TwoDLib::Redistribution> > FixtureAdapter::Mapping(const std::string& type) const
{
     pugi::xml_document doc;
     std::vector< std::vector<TwoDLib::Redistribution> > vec_vec_rev;

     pugi::xml_parse_result result0 = doc.load_file("condee2a5ff4-0087-4d69-bae3-c0a223d03693.model");
     pugi::xml_node  root0 = doc.first_child();
     std::vector<TwoDLib::Redistribution> vec_map = TwoDLib::RetrieveMappingFromXML(type,root0);
     std::vector< std::vector<TwoDLib::Redistribution> > vec_vec_map { vec_map };
     return vec_vec_map;
}

std::unique_ptr<TwoDLib::Ode2DSystemGroup> FixtureAdapter::Group() const
{
       std::unique_ptr<TwoDLib::Ode2DSystemGroup> pret(new TwoDLib::Ode2DSystemGroup(_vec_mesh,_vec_vec_rev,_vec_vec_res));
       pret->Initialize(0,0,0);
       return pret;
}

FixtureAdapter::FixtureAdapter():
_vec_mesh(this->Mesh()),
_vec_vec_rev(this->Mapping("Reversal")),
_vec_vec_res(this->Mapping("Reset")),
_pgroup(Group())
{
}

FixtureAdapter::~FixtureAdapter()
{
}
