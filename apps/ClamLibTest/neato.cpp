#ifdef WIN32
#pragma warning(disable: 4996)
#endif 

#include <iostream>
#include <string>
#include <sstream>
#include <DynamicLib/DynamicLib.h>
#include <UtilLib/UtilLib.h>
#include <ClamLib/ClamLib.h>
#include <TApplication.h>

using std::string;
using ClamLib::CircuitDescription;
using ClamLib::ForwardSubNetworkIterator;
using ClamLib::Id;
using ClamLib::NeatoAttributes;
using ClamLib::ReverseSubNetworkIterator;
using ClamLib::SimulationResult;
using ClamLib::SimulationResultIterator;
using DynamicLib::D_DynamicNetwork;
using DynamicLib::D_DynamicNode;
using UtilLib::GeneralException;
 
const string PATH = "C:/cygwin/home/Marc/tmp";
const string NET  = "ongabonga.net";
const string DATA = "data.root";
const string EXT  = ".neato";
const string NEATO = "woef.dot";

const float OFFSET_INPUT			= 16.0;
const float POS_FBK					= 4800.0;
const float LAYER_SCALE_FACTOR		= 1200.0;
const float FEATURE_OFFSET			= 1200.0;
const float NODE_SIZE				= 0.08F;

const float OFFSET_INHIB_X			= 0.0;
const float OFFSET_INHIB_Y			= -3600.0;
const float DIS_FEATURE_OFFSET		= 1000.0;
const float POS_SCALE_FACTOR_DIS	= 90.0;
const float DIS_CIRCUIT_SIZE		= 20.0;

const PhysicalPosition MASK = { 2, 2, 2, 0};

bool IsMaskFit(const PhysicalPosition& p1, const PhysicalPosition& p2)
{
	return 
		( 
			(p1._position_x == p2._position_x) && 
			(p1._position_y == p2._position_y) && 
			(p1._position_z == p2._position_z) &&
			(p1._position_depth == p2._position_depth) ) ? true : false;
}

struct Role {
	vector<float> _x;
	vector<float> _y;
};

struct Range {
	string	_name;
	Id		_min;
	Id		_max;

	bool InRange(Id id) const { return ((id._id_value >= _min._id_value)&& (id._id_value <= _max._id_value) ? true: false);}
};

struct IdPos {
	Id		_id;
	float	_x;
	float	_y;
};

bool operator==(const IdPos& idpos, Id id){
	return ( idpos._id == id);
}

Point GlobalPos(const PhysicalPosition& pos, const NeatoAttributes att, Number n_layer)
{
	Point p;
	double zpos;

	if (att._alignment == 1)
		zpos = pos._position_z;
	else
		zpos = n_layer - pos._position_z - 1;

	p._x = att._global_offset._x + att._node_translation_vector._x*pos._position_x + pos._position_depth*att._feature_translation_vector._x + att._layer_scale_factor._x*zpos; 
	p._y = att._global_offset._y + att._node_translation_vector._y*pos._position_y + pos._position_depth*att._feature_translation_vector._y;

	return p;
}

void AssignDefaultPositions
(
	SimulationResultIterator	iter, 
	const NeatoAttributes&		att,
	vector<IdPos>*				pvec_pos,
	Range*						p_range,
	vector<Index>*				pvec_id,
	vector<Index>*				pvec_mask,
	const PhysicalPosition		mask
 )
{
	Id max = Id(0);
	Id min = Id(1000000);
	// This is the standard JOCN net lay-out, so there should be 6 populations in the CircuitNodeRole
	CircuitDescription desc = iter->GetCircuitDescription();
	Number n_circ = desc.RoleVec().size();
	if ( n_circ != 6){
		cout << "This is a not a perceptron circuit" << endl;
		exit(0);
	}

	vector<float> x(n_circ);
	vector<float> y(n_circ);

	// How many layers ?
	Number n_layer = iter->begin().LayeredNetDescription().NumberOfLayers();
	double signum = att._alignment;

	x[0] = 0.0;
	y[0] = att._perc_offset._y;
	x[1] = 0.0F;
	y[1] = -att._perc_offset._y;
	x[2] = -signum*att._perc_offset._x;
	y[2] = 1.5*att._perc_offset._y;
	x[3] = -signum*att._perc_offset._x;
	y[3] = 0.5*att._perc_offset._y;
	x[4] = -signum*att._perc_offset._x;
	y[4] = - 0.5*att._perc_offset._y;
	x[5] = -signum*att._perc_offset._x;
	y[5] = - 1.5*att._perc_offset._y;

	ForwardSubNetworkIterator fsniter_b = iter->begin();
	ForwardSubNetworkIterator fsniter_e = iter->end();
	PhysicalPosition pos;
	for
	(
		ForwardSubNetworkIterator iter = fsniter_b;
		iter != fsniter_e;
		iter++
	)
	{
		iter.Position(&pos);
		Point p = GlobalPos(pos,att,n_layer);

		for (Index i = 0; i < n_circ; i++)
		{
			Id id = (*iter)[i];
			IdPos idpos;
			idpos._id = id;
			idpos._x  = p._x + x[i];
			idpos._y  = p._y + y[i];
			pvec_pos->push_back(idpos);
			(*pvec_id)[id._id_value] = 1;
			max = (id._id_value > max._id_value) ? id : max;
			min = (id._id_value < min._id_value) ? id : min;
			if ( IsMaskFit(pos,mask) )
				pvec_mask->push_back(id._id_value);		

		}
	}
	p_range->_min = min;
	p_range->_max = max;
}

void AssignDISINHIBPositions
(
	SimulationResultIterator	iter, 
	const NeatoAttributes&		att,
	vector<IdPos>*				pvec_pos,
	vector<Index>*				pvec_id,
	vector<Index>*				pvec_mask,
	const PhysicalPosition&		mask
)
{

	// fantastic, we want this file
	CircuitDescription desc = iter->GetCircuitDescription();
	Number n_disinhib  = desc.RoleVec().size();

	vector<float> x(n_disinhib,OFFSET_INHIB_X);
	vector<float> y(n_disinhib,OFFSET_INHIB_Y);
	x[0] += -DIS_CIRCUIT_SIZE;
	y[0] += +DIS_CIRCUIT_SIZE;
	x[1] += +DIS_CIRCUIT_SIZE;
	y[1] += -DIS_CIRCUIT_SIZE;
	x[2] +=  -1.5*DIS_CIRCUIT_SIZE;
	y[2] += 1.5*DIS_CIRCUIT_SIZE;
	x[3] += +DIS_CIRCUIT_SIZE;
	y[3] += +DIS_CIRCUIT_SIZE;
	x[4] += -DIS_CIRCUIT_SIZE;
	y[4] += -DIS_CIRCUIT_SIZE;
	x[5] += 1.5*DIS_CIRCUIT_SIZE;
	y[5] += 1.5*DIS_CIRCUIT_SIZE;
	x[6] += 0.5*DIS_CIRCUIT_SIZE;
	y[6] += 0.5*DIS_CIRCUIT_SIZE;
	x[7] += -0.5*DIS_CIRCUIT_SIZE;
	y[7] += -0.5*DIS_CIRCUIT_SIZE;

	for (Index i = 0; i < n_disinhib; i++)
		cout << i << " " << desc.RoleVec()[i].GetName() << " " << desc.RoleVec()[i].Position()._x << " " << desc.RoleVec()[i].Position()._y << " " << desc.RoleVec()[i].Position()._z << " " << desc.RoleVec()[i].Position()._f << endl;

	Number n_layer = iter->begin().LayeredNetDescription().NumberOfLayers();

	ForwardSubNetworkIterator fsniter_b = iter->begin();
	ForwardSubNetworkIterator fsniter_e = iter->end();
	PhysicalPosition pos;
	for
	(
		ForwardSubNetworkIterator iter = fsniter_b;
		iter != fsniter_e;
		iter++
	)
	{
		for (Index i = 0; i < n_disinhib; i++){
			iter.Position(&pos);

			Point p = GlobalPos(pos,att,n_layer);

			Id id = (*iter)[i];
			IdPos idpos;
			idpos._id = id;
			idpos._x  = p._x + x[i];
			idpos._y  = p._y + y[i];
			(*pvec_id)[id._id_value] = 1;			
			if ( IsMaskFit(pos,mask) )
				pvec_mask->push_back(id._id_value);
			pvec_pos->push_back(idpos);
		}
	}
}

void AssignNodePositions
(
	SimulationResultIterator	iter, 
	vector<IdPos>*				pvec_pos,
	vector<Range>*				pvec_range,
	vector<Index>*				pvec_id,
	vector<Index>*				pvec_mask,
	const PhysicalPosition&		mask
)
{
	string fn = iter->GetName();
	cout << "AssignNodePositions: " << iter->GetName() << endl;
	
	NeatoAttributes att;

	Range range;

	if (fn == "gaussianffd"){
		range._name = fn;
		ParseNeatoAttributeFile
		(
			fn,
			&att
		);

		AssignDefaultPositions
		(
			iter, 
			att,
			pvec_pos,
			&range,
			pvec_id,
			pvec_mask,
			mask
		);
		pvec_range->push_back(range);
	}

	if (fn == "gaussiandisinhiblayerffd" ){
		range._name = fn;
		ParseNeatoAttributeFile
		(
			fn,
			&att
		);
		AssignDISINHIBPositions
		(
			iter,
			att,
			pvec_pos,
			pvec_id,
			pvec_mask,mask
		);
	}

	if (fn == "gaussianfbk"){
		range._name = fn;
		ParseNeatoAttributeFile
		(
			fn,
			&att
		);
		AssignDefaultPositions
		(
			iter, 
			att,
			pvec_pos,
			&range,
			pvec_id,
			pvec_mask,
			mask
		);
		pvec_range->push_back(range);
	}

/*

	if (fn == "gaussian_colourffd"){
		ParseNeatoAttributeFile
		(
			fn,
			&att
		);

		AssignDefaultPositions
		(
			iter,
			att,
			pvec_pos,
			&range,
			pvec_id,
			pvec_mask,
			mask

		);
	}


	if (fn == "gaussian_colourfbk"){
		ParseNeatoAttributeFile
		(
			fn,
			&att
		);

		AssignDefaultPositions
		(
			iter,
			att,
			pvec_pos,
			&range,
			pvec_id,
			pvec_mask,
			mask
		);
	}*/
}


void ProcessNeatoConfigFile
(
	SimulationResultIterator	iter, 
	const vector<IdPos>&		vec_pos,
	ofstream&					ofst
)
{
	cout << "ProcessNeatoConfigFile" << endl;
	static bool first = false;
	if (! first ){
		first = true;
		ofst <<"graph {\n";
	}
	
	BOOST_FOREACH(const IdPos& pos, vec_pos){
		ofst << "n" << pos._id << " [pos=\"" <<  pos._x << "," << pos._y << "!\",shape=circle,label=\"\",width=" << NODE_SIZE << ",fixedsize=true];\n";
	}
}

void MopUp
(
	D_DynamicNetwork&		d_net,
	const vector<IdPos>&	vec_pos,
	const vector<Range>&	vec_range,
	const vector<Index>&	vec_id,
	const vector<Index>&	vec_mask,
	ofstream&				ofst
 )
{

	for
	(
		vector<IdPos>::const_iterator iter = vec_pos.begin();
		iter != vec_pos.end();
		iter++
	)
	{
		for
		(
			D_DynamicNetwork::predecessor_iterator piter = d_net.begin(NodeId(iter->_id._id_value));
			piter != d_net.end(NodeId(iter->_id._id_value));
			piter++
		)
		{
			// if this node or its predecessor is in the masking vector, write out the connection, but only if both nodes
			// are visualized
			vector<Index>::const_iterator iter_parent;
			vector<Index>::const_iterator iter_child;
			iter_parent = std::find(vec_mask.begin(),vec_mask.end(),iter->_id._id_value);
			iter_child  = std::find(vec_mask.begin(),vec_mask.end(),piter->MyNodeId()._id_value);
			
			double weight = piter.GetWeight();
			if ( ((iter_parent != vec_mask.end() ) || (iter_child != vec_mask.end() )) && vec_id[iter->_id._id_value] == 1 && vec_id[piter->MyNodeId()._id_value] == 1 && weight > 1e-5) 
				ofst << "n" << iter->_id << " --  n" << piter->MyNodeId() << " [penwidth=1,color=\"#EEAAAA\"];\n";
		}
	} 
	ofst << "}\n";
	cout << "Mop up finished" << endl;
}

int main(int argc, char *argv[])
{
	TApplication theApp("theApp", &argc, argv);
	const string netpath = PATH + "/" + NET;
	const string rootpath = PATH + "/"  + DATA;	
	ifstream ifst(netpath.c_str());

	if (ifst == 0){
	  cout << "Couldn't open file: " << netpath.c_str() << endl;
	  exit(0);
  }

	try {
		D_DynamicNetwork net;
		net.FromStream(ifst);

		cout << net.NumberOfNodes() << " nodes in network" << endl;


		TFile file(rootpath.c_str());
		SimulationResult res(file);

	
		ofstream ofst(NEATO.c_str());
		vector<IdPos> vec_pos;
		vector<Range> vec_range;
		vector<Index> vec_id(net.NumberOfNodes(),0);
		vector<Index> vec_mask;

		for (SimulationResultIterator iter = res.begin(); iter != res.end(); iter++)
		{
			AssignNodePositions(iter,&vec_pos,&vec_range,&vec_id,&vec_mask,MASK);
			ProcessNeatoConfigFile(iter,vec_pos,ofst);
		}

		MopUp(net,vec_pos,vec_range,vec_id,vec_mask, ofst);
	} // try
	
	catch(const GeneralException& exec){
		cout << exec.Description() << endl;
	}

	return 0;
}
