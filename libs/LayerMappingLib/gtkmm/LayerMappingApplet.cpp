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

#include "LayerMappingApplet.h"

using namespace LayerMappingLib::gtkmm;

LayerMappingApplet::LayerMappingApplet() : _vbox( false ),
	_table( 1, 2, false ),
	_box_button( false ),
	_button_evolve( "Evolve" ),
	_button_set_feedback_activation( "Set Feedback Activation" )
{
	_button_evolve.signal_clicked().connect( sigc::mem_fun( *this, &LayerMappingApplet::on_button_evolve_clicked ) );

	_button_set_feedback_activation.signal_clicked().connect( sigc::mem_fun( *this, &LayerMappingApplet::on_button_set_feedback_activation_clicked ) );

	_box_button.pack_start( _button_evolve );
	_box_button.pack_start( _button_set_feedback_activation );
	_button_evolve.show();
	_button_set_feedback_activation.show();
	_box_button.show();

	_table.attach( _box_button,
		0, 1,
		0, 1,
		Gtk::SHRINK,
		Gtk::SHRINK );
	_table.attach( _ensemble_widget,
		1, 2,
		0, 1 );
	_table.show();



	//File menu:
	{
		Gtk::Menu::MenuList& menulist = _filemenu.items();
		
		menulist.push_back( Gtk::Menu_Helpers::MenuElem("_Load Image",  Gtk::AccelKey("<control>l"),
		sigc::mem_fun(*this, &LayerMappingApplet::on_filemenu_load ) ) );

		menulist.push_back( Gtk::Menu_Helpers::MenuElem("_Quit",  Gtk::AccelKey("<control>q"),
		sigc::mem_fun(*this, &LayerMappingApplet::on_filemenu_quit ) ) );
	}

	_menubar.items().push_back( Gtk::Menu_Helpers::MenuElem("_File", _filemenu ) );

	_vbox.pack_start( _menubar, Gtk::PACK_SHRINK );
	_vbox.pack_start( _table );
	_vbox.show();

	Gtk::Window::add( _vbox );

	Gtk::Window::resize( 800, 600 );

	show_all_children();
}

void LayerMappingApplet::setNetworkEnsemble( NetworkEnsemble<double>& network_ensemble )
{
	_network_ensemble = network_ensemble;

	for( NetworkEnsemble<double>::iterator i = _network_ensemble.begin();
		i != _network_ensemble.end();
		i++ )
	{
		_ensemble_widget.add( *i );
	}

// 	ImageConverter::sublayer( _image, *_network->input_activation() );
}

void LayerMappingApplet::set_feedback_vector( vector<double>& v )
{
	_feedback_vector = v;
}

void LayerMappingApplet::show()
{
	Gtk::Window::show();

	_box_button.show();
	_button_evolve.show();
	_button_set_feedback_activation.show();
	_ensemble_widget.show();
	_table.show();
}

// Gtk::Image* LayerMappingApplet::loadImage( string s )
// {
// // 	_image.set( s );
// 
// // 	return &_image;
// }

void LayerMappingApplet::on_filemenu_load()
{
	cout << "Loading file... later" << endl;
	Gtk::FileChooserDialog chooser(  "Select an Image" );

	chooser.show();
}

void LayerMappingApplet::on_filemenu_quit()
{
	hide();
}

void LayerMappingApplet::_redraw()
{
	Glib::RefPtr<Gdk::Window> win = get_window();
	if( win )
	{
		Gdk::Rectangle r( 0, 0,
			get_allocation().get_width(),
			get_allocation().get_height() );
		win->invalidate_rect( r, true );
	}
}

void LayerMappingApplet::on_button_evolve_clicked()
{
	cout << "Evolving..." << endl;
	int i = 0;
	for( NetworkEnsemble<double>::iterator iter = _network_ensemble.begin();
		iter != _network_ensemble.end();
		iter++, i++ )
	{
		iter->update();
// 		cout << "# " << i << endl;
	}

	cout << "done" << endl;

	_redraw();
}

void LayerMappingApplet::on_button_set_feedback_activation_clicked()
{
	NetworkEnsemble<double>::network feedback = *( ++_network_ensemble.begin() );
	cout << "This function is for debug porposes only. Application may crash if used with a wrong network ensemble." << endl;

	int layer = 0;
	vector<double>::iterator fb = _feedback_vector.begin();
	for( NetworkEnsemble<double>::network::iterator i = feedback.begin( layer );
		i != feedback.end( layer );
		i++, fb++ )
	{
		double t[ 1 ];
		*t = *fb;
// 		fill( t, t + i->width() * i->height(), 1 );
		i->activation().get( t );
	}
	assert( fb == _feedback_vector.end() );

	_redraw();
}
