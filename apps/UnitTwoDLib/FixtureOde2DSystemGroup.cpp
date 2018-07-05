#include "FixtureOde2DSystemGroup.hpp"

using namespace TwoDLib;

bool FixtureOde2DSystemGroup::CreateMeshFile()
{
	// create two simple meshes
	std::ofstream strmesh1("mesh1.mesh");
	strmesh1 << "0.0\n";
	strmesh1 << "1.0\n";
	strmesh1 << "0.0 1.0 2.0\n";
	strmesh1 << "1.0 1.0 1.0\n";
	strmesh1 << "0.0 1.0 2.0 3.0\n";
	strmesh1 << "2.0 2.0 2.0 2.0\n";
	strmesh1 << "0.0 1.0 2.0 3.0\n";
	strmesh1 << "3.0 3.0 3.0 3.0\n";
	strmesh1 << "closed\n";
	strmesh1 << "end\n";
	strmesh1.close();
	// create two simple meshes
	std::ofstream strmesh2("mesh2.mesh");
	strmesh2 << "0.0\n";
	strmesh2 << "1.0\n";
	strmesh2 << "0.0 1.0 2.0 3.0\n";
	strmesh2 << "1.0 1.0 1.0 1.0\n";
	strmesh2 << "0.0 1.0 2.0 3.0\n";
	strmesh2 << "2.0 2.0 2.0 2.0\n";
	strmesh2 << "0.0 1.0 2.0\n";
	strmesh2 << "3.0 3.0 3.0\n";
	strmesh2 << "closed\n";
	strmesh2 << "end\n";
	strmesh2.close();

	return true;
}

FixtureOde2DSystemGroup::FixtureOde2DSystemGroup():
_mesh_create(this->CreateMeshFile()),
_mesh1("mesh1.mesh"),
_mesh2("mesh2.mesh")
{
	vector<double> vstat1{ 0., 1., 1., 0.};
	vector<double> wstat1{ 0., 0., 1., 1.};
	vector<double> vstat2{ 1., 2., 2., 1.};
	vector<double> wstat2{ 0., 0., 1., 1.};
	Quadrilateral stat1(vstat1,wstat1);
	Quadrilateral stat2(vstat2,wstat2);
	_mesh1.InsertStationary(stat1);
	_mesh1.InsertStationary(stat2);

	_mesh2.InsertStationary(stat1);
}

FixtureOde2DSystemGroup::~FixtureOde2DSystemGroup()
{
}
