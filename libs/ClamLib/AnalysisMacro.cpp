// Copyright (c) 2005 - 2009 Marc de Kamps
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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
/*
#ifdef WIN32
#pragma warning(disable: 4244)
#pragma warning(disable: 4267)
#pragma warning(disable: 4800)
#endif

#include <iostream>
#include <string>
#include <sstream>
#include <TFile.h>
#include <TNtuple.h>
#include <TCanvas.h>
#include <TGraph.h>
#include <TMatrix.h>
#include <TPad.h>
#include <TColor.h>
#include "ClamLibPositions.h"

using namespace std;

// adapt this to your local directory
string JOCN_ROOT_FILE_PATH("Z:/clamdevelop/code/apps/CLAM_Win32/test/");
string ROOT_FILE("jocn_disinhibition.root");
TCanvas* P_CANVAS;
TPad*    PAD_VENTRAL;

TFile*   OpenRootFile(const string&);
TNtuple* GetInfoTuple(TFile*);
void     ProcessTuple(TNtuple&);
void     CreateMappings(TNtuple&);
void     ShowVentralFFDPlane(Int_t, Int_t, Double_t);
void     ShowVentralREVPlane(Int_t, Int_t, Double_t);
TGraph*  GraphFromPosition(Float_t, Float_t, Float_t, Float_t);
Int_t	 ConvertPositionToNeuronNumber(Float_t, Float_t, Float_t, Float_t);
Int_t	 ConvertPositionToNeuronNumber(TNtuple&, Float_t, Float_t, Float_t, Float_t);
TMatrix  GetPositiveFFDVentralPlane(Int_t, Int_t, Double_t, TFile&);
TMatrix  GetNegativeFFDVentralPlane(Int_t, Int_t, Double_t, TFile&);
TMatrix	 GetPositiveREVVentralPlane(Int_t, Int_t, Double_t, TFile&);
TMatrix	 GetNegativeREVVentralPlane(Int_t, Int_t, Double_t, TFile&);
bool     IsVentralFFD(Float_t, Float_t, Float_t);
bool     IsVentralREV(Float_t, Float_t, Float_t);
TGraph*  GetGraphFromId(Int_t);
TGraph*  GetGraphFromId(Int_t, TFile&);

//Less than ideal these constant must be maintained as in ...
Float_t P_POP_VENTRAL_X_OFFSET = 0.05F;
Float_t P_POP_VENTRAL_Y_OFFSET = 0.0F;
Float_t P_POP_VENTRAL_Z_OFFSET = 0.10F;
Float_t MAX_VENTRAL_LAYER_SIZE = 24;
Float_t FEEDBACK_Y_OFFSET = 50.0F;

struct RGB {
	Float_t _r;
	Float_t _g;
	Float_t _b;
};

enum Sign {POSITIVE, NEGATIVE };

const RGB RGB_POSITIVE = { 255.0F, 0.0F, 0.0F };
const RGB RGB_NEGATIVE = { 0.0F, 255.0F, 0.0F };
const Float_t ALPHA = 0.2F;


RGB operator*(const RGB& col, Float_t val)
{
	RGB col_ret = col;
	col_ret._r *= val;
	col_ret._g *= val;
	col_ret._b *= val;

	return col_ret;
}

void SetRootFile(const char* p_char)
{
	string str(p_char);
	ROOT_FILE = str;

	TFile* p_file = OpenRootFile(JOCN_ROOT_FILE_PATH+ROOT_FILE);
	if (p_file)
		cout << "New root file is: " << ROOT_FILE << ". Opened succesfully." << endl;
	else
		cout << "New root file is: " << ROOT_FILE << ". Opened unsuccesfully." << endl;

	p_file->Close();
}

void JOCNAnalysis()
{
	TFile* p_file = OpenRootFile(JOCN_ROOT_FILE_PATH+ROOT_FILE);

	TNtuple* p_tuple = GetInfoTuple(p_file);

	ProcessTuple(*p_tuple);
}

TFile* OpenRootFile(const string& file_name)
{
	TFile* p_file = new TFile(file_name.c_str());
	return p_file;
}

TNtuple*  GetInfoTuple(TFile* p_file)
{
	TNtuple* p_tuple = (TNtuple*)p_file->Get("infotuple");
	return p_tuple;
}

void ProcessTuple(TNtuple& tuple)
{
}

TGraph* GraphFromPosition(Float_t x, Float_t y , Float_t z, Float_t f)
{
	TFile* p_file = OpenRootFile(JOCN_ROOT_FILE_PATH+ROOT_FILE);

	TNtuple* p_tuple = GetInfoTuple(p_file);

	Int_t n_neuron = ConvertPositionToNeuronNumber(*p_tuple, x, y, z, f);

	ostringstream str;
	str << "rate_" << n_neuron;

	cout << str.str() << endl;

	TGraph* p_graph = (TGraph*)p_file->Get(str.str().c_str());
	return p_graph;
}



RGB ConvertToPositiveColour(Float_t val)
{

	if (val < 0)
		cout << "Colour value negative !!" << endl;

	RGB col_ret = RGB_POSITIVE*pow(val,ALPHA);

	return col_ret;
}

RGB ConvertToNegativeColour(Float_t val)
{
	if (val < 0)
		cout << "Colour value negative" << endl;

	RGB col_ret = RGB_NEGATIVE*pow(val,ALPHA);

	return col_ret;
}


void ProjectMatrixOnPad(TPad* p_pad, const TMatrix& mat, Sign sign)
{
	Int_t n_cols = mat.GetNcols();
	Int_t n_rows = mat.GetNrows();

	p_pad->SetFillColor(1);

	if ( p_pad->GetPad(1) == 0)
	{
		// no subdivisions
		p_pad->Divide(n_cols,n_rows,1e-3F,1e-3F);
		p_pad->Draw();
	}
	else
	{
		// pad doesn't have to be subdivided
		// number of subpads should match number of matrix entries
	}

	Int_t i_label = 0;

	for (Int_t i_col = 0; i_col < n_cols; i_col++)
		for (Int_t i_row = 0; i_row < n_rows; i_row++)
		{
			p_pad->cd(++i_label);
			RGB col;
			col._r = 0;
			col._g = 0;
			col._b = 0;

			if (sign == POSITIVE )
			{
				col = ConvertToPositiveColour(mat(i_col,i_row));
			}
			else
			{
				if ( sign == NEGATIVE )
				{
					col = ConvertToNegativeColour(mat(i_col,i_row));
				}
				// else
				   // do nothing (colour eq 0)
			}

			
			TColor* p_color = new TColor;
			Int_t col_ind =  p_color->GetColor((Int_t)col._r,(Int_t)col._g,(Int_t)col._b);
			gPad->SetFillColor(col_ind);
			if (col._r != 0 || col._g != 0 || col._b != 0)
				gPad->Draw();
			

		}

}

void ShowVentralFFDPlane(Int_t n_layer, Int_t n_feature, Double_t time)
{


	string absolute_path = JOCN_ROOT_FILE_PATH + ROOT_FILE;

	TFile* p_file = OpenRootFile(absolute_path);
	TNtuple* p_tuple = GetInfoTuple(p_file);

	TMatrix mat_pos = GetPositiveFFDVentralPlane(n_layer, n_feature, time, *p_file);
	TMatrix mat_neg = GetNegativeFFDVentralPlane(n_layer, n_feature, time, *p_file);

	PAD_VENTRAL = new TPad("p","",0.,0.,0.95,0.95);

	ProjectMatrixOnPad(PAD_VENTRAL,mat_pos,POSITIVE);
	ProjectMatrixOnPad(PAD_VENTRAL,mat_neg,NEGATIVE);


}

void ShowVentralREVPlane(Int_t n_layer, Int_t n_feature, Double_t time)
{

	string absolute_path = JOCN_ROOT_FILE_PATH + ROOT_FILE;

	TFile* p_file = OpenRootFile(absolute_path);
	TNtuple* p_tuple = GetInfoTuple(p_file);

	TMatrix mat_pos = GetPositiveREVVentralPlane(n_layer, n_feature, time, *p_file);
	TMatrix mat_neg = GetNegativeREVVentralPlane(n_layer, n_feature, time, *p_file);

	PAD_VENTRAL = new TPad("p","",0.,0.,0.95,0.95);

	ProjectMatrixOnPad(PAD_VENTRAL,mat_pos,POSITIVE);
	ProjectMatrixOnPad(PAD_VENTRAL,mat_neg,NEGATIVE);


}

void DetermineFFDLayerSize(Int_t n_layer, Int_t* p_cols, Int_t* p_rows, TNtuple& tuple)
{
	*p_cols = *p_rows = 0;

	Int_t n_events = tuple.GetEntries();

	Float_t x_pos, y_pos, z_pos, f_pos;

	tuple.SetBranchAddress("x",&x_pos);
	tuple.SetBranchAddress("y",&y_pos);
	tuple.SetBranchAddress("z",&z_pos);
	tuple.SetBranchAddress("f",&f_pos);

	for ( int i = 0; i < n_events; i++)
	{
		tuple.GetEntry(i);

		if ( n_layer == floor(z_pos) && IsVentralFFD(x_pos, y_pos, z_pos) )
		{
			if ( floor(x_pos) > *p_cols)
				*p_cols = x_pos;
			if ( floor(y_pos) > *p_rows)
				*p_rows = y_pos;
		}
	}
	(*p_cols)++;
	(*p_rows)++;

}

void DetermineREVLayerSize(Int_t n_layer, Int_t* p_cols, Int_t* p_rows, TNtuple& tuple)
{
	*p_cols = *p_rows = 0;

	Int_t n_events = tuple.GetEntries();

	Float_t x_pos, y_pos, z_pos, f_pos;

	tuple.SetBranchAddress("x",&x_pos);
	tuple.SetBranchAddress("y",&y_pos);
	tuple.SetBranchAddress("z",&z_pos);
	tuple.SetBranchAddress("f",&f_pos);

	for ( int i = 0; i < n_events; i++)
	{
		tuple.GetEntry(i);

		if ( n_layer == floor(z_pos) && IsVentralREV(x_pos, y_pos, z_pos) )
		{
			if ( floor(x_pos) > *p_cols)
				*p_cols = x_pos;
			if ( floor(y_pos) > *p_rows)
				*p_rows = y_pos;
		}
	}
	(*p_cols)++;
	(*p_rows)++;
	(*p_rows) -= FEEDBACK_Y_OFFSET;

}

TGraph* GetGraphFromId(Int_t n_neuron)
{
	string absolute_path = JOCN_ROOT_FILE_PATH + ROOT_FILE;

	TFile* p_file = OpenRootFile(absolute_path);

	return GetGraphFromId(n_neuron,*p_file);
}

TGraph* GetGraphFromId(Int_t n_neuron, TFile& file)
{
	ostringstream ostr;
	ostringstream str;
	str << "rate_" << n_neuron;

	TGraph* p_ret = (TGraph*)file.Get(str.str().c_str());


	if ( ! p_ret)
		cout << "Couldn't get graph from file" << endl;


	return p_ret;
}

bool StripRateFromGraph(TGraph* p_graph, Float_t time, Float_t* p_rate)
{
	
	Int_t n_points = p_graph->GetN();

	Double_t t_min, rate_min;
	p_graph->GetPoint(0,t_min,rate_min);

	Double_t t_max, rate_max;
	p_graph->GetPoint(n_points - 1, t_max, rate_max);
	if (time > t_max || time < t_min)
	{
		cout << "Interpolation error" << endl;
		return false;
	}

	Float_t delta_v = (t_max - t_min)/n_points;
	Int_t n_interpol = floor(( time - t_min)/delta_v);

	Double_t time_interpol, rate;
	p_graph->GetPoint(n_interpol, time_interpol, rate);

	*p_rate = rate;

	return true;
}

Float_t GetRate
(
	TFile& file,
	Float_t x,
	Float_t y,
	Float_t z,
	Float_t f,
	Float_t time
)
{
	TNtuple* p_tuple = GetInfoTuple(&file);
	Int_t n_neur = ConvertPositionToNeuronNumber(*p_tuple, x, y, z, f); 

	TGraph* p_graph = GetGraphFromId(n_neur,file);

	Float_t rate;
	bool b_ret = StripRateFromGraph(p_graph,time,&rate);

	if (! b_ret)
		cout << "interpolation error in GetRate" << endl;

	return rate;
}

TMatrix GetPositiveREVVentralPlane(Int_t n_layer, Int_t n_feature, Double_t time, TFile& file)
{

	Int_t n_cols, n_rows;
	TNtuple *p_tuple = GetInfoTuple(&file);


	DetermineREVLayerSize(n_layer,  &n_cols, &n_rows, *p_tuple);

	TMatrix mat(n_cols,n_rows);

	for (Float_t i_col = 0; i_col < n_cols; i_col++)
		for (Float_t i_row = 0; i_row < n_rows; i_row++)
		{
			mat(i_col, i_row) = GetRate
			(
				file, 
				i_col   + P_POP_VENTRAL_X_OFFSET, 
				i_row   + P_POP_VENTRAL_Y_OFFSET + FEEDBACK_Y_OFFSET,
				n_layer + P_POP_VENTRAL_Z_OFFSET,
				n_feature,
				time
			);
		}

	return mat;
}

TMatrix GetNegativeREVVentralPlane(Int_t n_layer, Int_t n_feature, Double_t time, TFile& file)
{
	Int_t n_cols, n_rows;
	TNtuple *p_tuple = GetInfoTuple(&file);

	DetermineFFDLayerSize(n_layer,  &n_cols, &n_rows, *p_tuple);

	TMatrix mat(n_cols,n_rows);

	for (Float_t i_col = 0; i_col < n_cols; i_col++)
		for (Float_t i_row = 0; i_row < n_rows; i_row++)
		{
			mat(i_col, i_row) = GetRate
			(
				file, 
				i_col   - P_POP_VENTRAL_X_OFFSET, 
				i_row   + P_POP_VENTRAL_Y_OFFSET + FEEDBACK_Y_OFFSET,
				n_layer + P_POP_VENTRAL_Z_OFFSET,
				n_feature,
				time
			);
		}

	return mat;
}

TMatrix GetPositiveFFDVentralPlane(Int_t n_layer, Int_t n_feature, Double_t time, TFile& file)
{
	Int_t n_cols, n_rows;
	TNtuple *p_tuple = GetInfoTuple(&file);

	DetermineFFDLayerSize(n_layer,  &n_cols, &n_rows, *p_tuple);

	TMatrix mat(n_cols,n_rows);

	for (Float_t i_col = 0; i_col < n_cols; i_col++)
		for (Float_t i_row = 0; i_row < n_rows; i_row++)
		{
			mat(i_col, i_row) = GetRate
			(
				file, 
				i_col   + P_POP_VENTRAL_X_OFFSET, 
				i_row   + P_POP_VENTRAL_Y_OFFSET,
				n_layer + P_POP_VENTRAL_Z_OFFSET,
				n_feature,
				time
			);
		}

	return mat;
}


TMatrix GetNegativeFFDVentralPlane(Int_t n_layer, Int_t n_feature, Double_t time, TFile& file)
{
	Int_t n_cols, n_rows;
	TNtuple *p_tuple = GetInfoTuple(&file);

	DetermineFFDLayerSize(n_layer,  &n_cols, &n_rows, *p_tuple);

	TMatrix mat(n_cols,n_rows);

	for (Float_t i_col = 0; i_col < n_cols; i_col++)
		for (Float_t i_row = 0; i_row < n_rows; i_row++)
		{
			mat(i_col, i_row) = GetRate
			(
				file, 
				i_col   - P_POP_VENTRAL_X_OFFSET, 
				i_row   + P_POP_VENTRAL_Y_OFFSET,
				n_layer + P_POP_VENTRAL_Z_OFFSET,
				n_feature,
				time
			);
		}

	return mat;
}



Int_t ConvertPositionToNeuronNumber(TNtuple& tuple, Float_t x, Float_t y, Float_t z, Float_t f)
{
	Int_t i_ret = -1;


	Int_t n_events = tuple.GetEntries();

	Float_t x_pos, y_pos, z_pos, f_pos;

	tuple.SetBranchAddress("x",&x_pos);
	tuple.SetBranchAddress("y",&y_pos);
	tuple.SetBranchAddress("z",&z_pos);
	tuple.SetBranchAddress("f",&f_pos);

	for ( int i = 0; i < n_events; i++)
	{
		tuple.GetEntry(i);

		if ( x == x_pos && y == y_pos && z == z_pos && f == f_pos)
			if ( i_ret == -1 )
			{
				i_ret = i;
			}
			else
			{
				i_ret = -2;
			}
		// else
		//     carry on
	}
	return i_ret;
}

bool ConvertNeuronToPosition(Int_t i, Float_t* p_x, Float_t* p_y, Float_t* p_z, Float_t* p_f)
{
	TFile* p_file = OpenRootFile(JOCN_ROOT_FILE_PATH+ROOT_FILE);

	TNtuple* p_tuple = GetInfoTuple(p_file);


	p_tuple->SetBranchAddress("x",p_x);
	p_tuple->SetBranchAddress("y",p_y);
	p_tuple->SetBranchAddress("z",p_z);
	p_tuple->SetBranchAddress("f",p_f);


	p_tuple->GetEntry(i);

	return true;
}

Int_t ConvertPositionToNeuronNumber(Float_t x, Float_t y, Float_t z, Float_t f)
{
	TFile* p_file = OpenRootFile(JOCN_ROOT_FILE_PATH+ROOT_FILE);

	TNtuple* p_tuple = GetInfoTuple(p_file);


	return ConvertPositionToNeuronNumber(*p_tuple, x, y, z, f);
}

bool IsVentralFFD(Float_t x, Float_t y, Float_t z)
{
	return (y < MAX_VENTRAL_LAYER_SIZE && z < MAX_VENTRAL_LAYER_SIZE) ? true: false;
}

bool IsVentralREV(Float_t x, Float_t y, Float_t z)
{
	return (y > FEEDBACK_Y_OFFSET - 1) ? true : false;
}*/
