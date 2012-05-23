"""Wrapper for the HMAX implementation with LayerMappingLib.
        author: Johannes Drever
        year: 2008"""

import numpy

import LayerMapping

class HMAXFeedback:
	"""Wrapper for the HMAX implementation with LayerMappingLib. """
	def __init__( self, image, ( c1_pooling, c2_pooling ) = ( LayerMapping.MAX, LayerMapping.MAX ), padding_noise = 0.0 ):
		"""Build HMAX with feedback mechanism. Read in the image and induce activity in forward path."""
		( width, height ) = image.shape
		
		self.hmax_feedback = LayerMapping.Models.HMAX_Feedback( width, height, c1_pooling, c2_pooling )
		
		self.hmax_feedback.set_input( image.reshape( width * height ) )

		self.hmax_feedback.fill_feature_map_padding_with_noise( padding_noise )
		
		self.hmax_feedback.evolve( 0 )

	def fill_feature_map_padding_width_noise( level ):
		"""Fill the feature map padding with noise."""
		self.hmax_feedback.fill_feature_map_padding_with_noise( level )
		
	def feature_vector( self ):
		"""Return the feature vector of the HMAX model"""
		return self.hmax_feedback.feature_vector()
	
	def feature_map( self, feature_map, layer, network ):
		"""Return feature map in network ensemble"""
		( height, width ) =  self.hmax_feedback.feature_map_size( network, layer, feature_map )
		return numpy.array( self.hmax_feedback.feature_map( network, layer, feature_map ) ).reshape( ( width, height ) )
	
	def feedback( self, template ):
		"""Induce the attentional template"""
		self.hmax_feedback.set_feedback_template( template )
		self.hmax_feedback.evolve( 1 ) #feed the activity back
		self.hmax_feedback.evolve( 2 ) #perform the local consistency check
		self.hmax_feedback.evolve( 3 ) #process s2 mean and standard deviation
		
	def lcc_s2_mean( self, filter_band ):
		"""The mean of the S2 feature maps in the local consitency check network."""
		return self.feature_map( filter_band, 1, 3 )
		
	def lcc_s2_max( self, filter_band ):
		"""The max of the S2 feature maps in the local consistency check network."""
		return self.feature_map( filter_band, 3, 3 )
			
	def lcc_s2_min( self, filter_band ):
		"""The min of the S2 feature maps in the local consistency check network."""
		return self.feature_map( filter_band, 4, 3 )
	
	def lcc_s2_standard_deviation( self, filter_band ):
		"""The standard deviation of the S2 feature maps in the local consistency check network."""
		return self.feature_map( filter_band, 2, 3 )

	def input_image( self ):
		"""The input image."""
		return self.feature_map( 0, 0, 0 )

