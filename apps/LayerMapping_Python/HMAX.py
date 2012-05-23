"""Experimental implementation of HMAX according to Serre (2006).
	author: Johannes Drever
	year: 2008"""

import numpy

import LayerMapping
import Util
import pylab

serre_nr_s1_filter = [ 2, 2, 2, 2, 2, 2, 2, 2 ]

serre_s1_filter_size = range( 7, 39, 2 )

serre_s1_lambda = [ 3.5, 4.6, 5.6, 6.8, 7.9, 9.1, 10.3, 11.5, 12.7, 14.1, 15.4, 16.8, 18.2, 19.7, 21.2, 22.8 ]
serre_s1_sigma = [ 2.8, 3.6, 4.5, 5.4, 6.3, 7.3, 8.2, 9.2, 10.2, 11.3, 12.3, 13.4, 14.6, 15.8, 17.0, 18.2 ]
serre_s1_gamma = 0.3

serre_c1_receptive_field_size = [ 8, 10, 12, 14, 16, 18, 20, 22 ]
serre_c1_skip_size = [ 3, 5, 7, 8, 10, 12, 13, 15 ]

def filter_matrix_string( name, function_type, matrix ):
	"""Convert a filter matrix to a string in the suitable format for LayerMappingLib::FunctionFactory"""
	width, height = matrix.shape
	
	s = name + " " + function_type + ( " %d" % width ) + ( " %d" % height ) + " "

	for row in matrix:
		for x in row:
			s = s + "%f " % x

	return s

def convert_paches_to_function_string( patches, name = "s2", function_type = LayerMapping.CONVOLUTION ):
	"""Convert a list of patches into strings that can be interpreted as functions by LayerMappingLib::FunctionFactory."""
	r = []
	for i in range( len( patches ) ):
		r.append( filter_matrix_string( name, function_type, patches[ i ] ) )

	return r


def gabor( width, height, sigma, lambda_, theta, gamma ):
	"""Gabor filters."""
	r = numpy.zeros( ( width, height ) )

	for x in range( width ):
		for y in range( height ):
			x0 = x * numpy.cos( theta ) + y * numpy.sin( theta );
			y0 = -x * numpy.sin( theta ) + y * numpy.cos( theta );

			r[ x ][ y ] = numpy.exp( -( ( ( x0 * x0 ) + ( gamma * gamma * y0 * y0 ) ) / ( 2 * sigma * sigma ) ) ) * numpy.cos( ( 2 * numpy.pi ) / ( lambda_ ) * x0 )

	return r

def serre_s1_features( nr_orientations = 4 ):
	"""Gabor filters for S1 layer according to serre."""
	function_list = []
	for i in range( reduce( lambda x, y: x + y, serre_nr_s1_filter ) ):
		for theta in range( nr_orientations ):
			a = gabor( serre_s1_filter_size[ i ], serre_s1_filter_size[ i ], serre_s1_sigma[ i ], serre_s1_lambda[ i ], theta * numpy.pi / nr_orientations, serre_s1_gamma )
			
			function_list.append( convert_paches_to_function_string( [ a ], ( "C1_%dx%d" % ( serre_s1_filter_size[ i ], serre_s1_filter_size[ i ] ) + "_%d" % ( 360 / nr_orientations * theta ) ), LayerMapping.CONVOLUTION ) )

	return Util.flatten( function_list )

class HMAX:
	def __init__( self, image, s1_features = serre_s1_features(), s2_features = [ numpy.ones( ( 1, 1 ) ) ], ( c1_pooling, c2_pooling ) = ( LayerMapping.MAX, LayerMapping.MAX ), padding_noise = 0.0, c1_receptive_field_size = serre_c1_receptive_field_size, c1_skip_size = serre_c1_skip_size, nr_s1_filter = serre_nr_s1_filter ):
		( width, height ) = image.shape

		self.c1_receptive_field_size = c1_receptive_field_size
		self.c1_skip_size = c1_skip_size
		self.nr_s1_filter = nr_s1_filter
		self.nr_orientations = 4

		self.hmax = LayerMapping.Models.HMAX_Learned_S2( width, height, nr_s1_filter, s1_features, c1_pooling, c1_receptive_field_size, c1_skip_size, convert_paches_to_function_string( s2_features ), c2_pooling )
		
		self.hmax.set_input( image.reshape( width * height ) )

		self.hmax.fill_feature_map_padding_with_noise( padding_noise )
		
		self.hmax.evolve()

	def fill_feature_map_padding_width_noise( level ):
		self.hmax.fill_feature_map_padding_with_noise( level )
		
	def feature_vector( self ):
		return self.hmax.feature_vector()
	
	def feature_map( self, layer, feature_map ):
		( width, height ) = self.feature_map_size( layer, feature_map )
		return numpy.array( self.hmax.feature_map( 0, layer, feature_map ) ).reshape( ( height, width ) )
			
	def feature_map_size( self, layer, feature_map ):
		( width, height ) = self.hmax.feature_map_size( 0, layer, feature_map )
		return int( width ), int( height )

	def input_image( self ):
		return self.feature_map( 0, 0 )

	def nr_filter_bands( self ):
		return len( self.nr_s1_filter )

	def show_feature_map( self, layer, fm, figure = 1 ):
		pylab.imshow( self.feature_map( layer, fm ) )

	def C1_feature_maps( self, filter_band ):
		"""Return feature maps for all orientations at a specific scale and size"""
		print filter_band
		if filter_band > len( self.nr_s1_filter ):
			raise "filter band %d" + filter_band + " does not exists."

		fmi = filter_band * self.nr_orientations

		( width, height ) = self.feature_map_size( 2, fmi )
		size = height, width

		fm = []
		for i in range( self.nr_orientations ):
			fm.append( numpy.array( self.hmax.feature_map( 0, 2, fmi + i ) ).reshape( size ) )

		return fm

