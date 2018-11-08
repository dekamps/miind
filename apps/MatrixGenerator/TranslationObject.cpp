/*
 * TranslationObject.cpp
 *
 *  Created on: Apr 20, 2017
 *      Author: scsmdk
 */

#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include "TranslationObject.hpp"
#include "TwoDLibException.hpp"

using namespace TwoDLib;

// At this stage, we don't know what the Mesh looks like,
// but we know all jumps will be the same, and we save the
// values of that jump
TranslationObject::TranslationObject(double v, double w):_v(v),_w(w),_efficacy(v,w),_translation_list(0){}

TranslationObject::TranslationObject(std::istream& s):
_v(0),_w(0),
_efficacy(this->EfficacyFromJumpFile(s)),
_translation_list(this->ListFromDoc())
{
}

bool TranslationObject::GenerateTranslationList(const Mesh& mesh)
{

	if (_v != 0 || _w != 0 ){
		// this condition indicates that we have uniform jumps and that the translation list hasn't been built yet
		// we check that
		if (_translation_list.size() == 0){
			// all is well; use the mesh to instantiate the translation list
			for( unsigned int i = 0; i <  mesh.NrStrips(); i++){
				vector<Translation> vec_trans;
				for (unsigned int j = 0; j < mesh.NrCellsInStrip(i); j++){
					vec_trans.push_back(Translation(_v,_w));
				}
				_translation_list.push_back(vec_trans);
			}
		}
		else
			throw TwoDLibException("A uniform jump of size(0,0) has been chosen. This is not allowed.");
	} else {
		// OK, the list was loaded from the jump file, so the translation_list is already there;
		// we only have to check whether it's consistent with the mesh.
		for (unsigned int i = 0; i < mesh.NrStrips(); i++)
			if ( _translation_list[i].size() != mesh.NrCellsInStrip(i))
				throw TwoDLibException("The translation list and the mesh have different dimensions.");
	}
	return true;
}

Translation TranslationObject::EfficacyFromJumpFile(std::istream& s)
{
	pugi::xml_parse_result result = _doc.load(s);

	if (!result){
		std::cout << "Error description: " << result.description() << "\n";
		throw TwoDLib::TwoDLibException("XML parse for jmp file failed.");
	}

	if (std::string(_doc.first_child().name()) != "Jump" )
		throw TwoDLib::TwoDLibException("Jump tag expected");

	if (std::string(_doc.first_child().first_child().name()) != "Efficacy" )
		throw TwoDLib::TwoDLibException("Efficacy tag expected");

	std::istringstream isteff(_doc.first_child().first_child().child_value());
	// slightly odd construction, but it creates a string dummy1 without white spaces

	std::vector<double> vec_tr;
	double d;
	while(isteff >> d){
		vec_tr.push_back(d);
		if (isteff.peek() == ',' || isteff.peek() == ' ')
            isteff.ignore();
	}
	if (vec_tr.size() != 2)
		throw TwoDLibException("Efficacy does not have two arguments.");

	return Translation(vec_tr[0],vec_tr[1]);
}

std::vector< std::vector<Translation> > TranslationObject::ListFromDoc(){
	std::istringstream ifst(_doc.first_child().first_child().next_sibling().child_value());
	std::string cell;
	double vval, wval;

	std::vector<int> vec_cell(2,0);
    boost::char_separator<char> sep(",");
    int i_old = -1;

    std::vector<std::vector<Translation> > vec_ret(0);
    vector<Translation> vec_tr;
	while(ifst){

		ifst >> cell >> vval >> wval;
		if (! ifst.good())
			break;
		// add a new vector every time a new strip number appears
		boost::tokenizer<boost::char_separator<char>> tokensfr(cell, sep);
		std::transform(tokensfr.begin(),tokensfr.end(),vec_cell.begin(),boost::lexical_cast<double,std::string>);
		if (vec_cell[0] > i_old)
		{
			if (i_old > -1)
				vec_ret.push_back(vec_tr);
			i_old += 1;
			assert(vec_cell[0] == i_old);
			vec_tr = std::vector<Translation>(0);
		}
		vec_tr.push_back(Translation(vval,wval));
	}

	// don't forget to add the last vector
	vec_ret.push_back(vec_tr);
	return vec_ret;
}
