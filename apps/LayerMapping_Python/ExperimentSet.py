import random

import numpy
import pylab

import MyImage

import HMAXFeedback

#interpol = "nearest"
interpol = None

def collect_data_classify( file_name, positions = [ ( 5, 5 ), ( 100, 10 ), ( 7, 100 ), ( 107, 95 ) ], objects = [ "rectangle", "diamond", "triangle", "star", "circle", "object1", "object2", "object3", "object4", "object5" ], size = ( 160, 160 ), object_size = ( 50, 50 ), pen = ( 3, "white", 255 ) ):
	"""Collect C2 features of images containing objects at different positions."""
	f = open( file_name +  ".tab", 'w' )
	
	f.write( orange_header_feature_vector( 256, objects ) )
	
	i = 0
	for o1 in objects:
		for o2 in filter( lambda x: x != o1, objects ):
			for pos1 in positions:
					for pos2 in filter( lambda x: x != pos1, positions ):
						im = MyImage.MyImage( size )
						im.draw_object( ( o1, ( ( ( pos1, object_size ), pen ) ) ) )
						im.draw_object( ( o2, ( ( ( pos2, object_size ), pen ) ) ) )
						im.save( file_name + "/" + "image%d" % i )
						i += 1
						
						hmax = HMAXFeedback.HMAXFeedback( im.getarray() )
						
						for x in hmax.feature_vector():
							f.write( """%f	""" % x )
						
						for x in map( lambda x: ( o1 == x or o2 == x ) and 1.0 or 0.0, objects ):
							f.write( """%f	""" % x )
							
						f.write( "\n" )
						

			
def do_attention( object_properties_list, keep = True, pooling = ( "Max", "Max" ), size = ( 64, 64 ), feedback_threshold = 0.0, noise = 0.0 ):
	if len( object_properties_list ) < 2:
		raise "You need at least two objects if you want to attent to one of them."

	hmax_feedback_list = []

	for i in range( len( object_properties_list ) ):
		image = MyImage.MyImage( size )

		object_properties_list[ i ] = image.draw_object( object_properties_list[ i ] )
	
		attention_feature_vector = HMAXFeedback.HMAXFeedback( image.getarray(), pooling ).feature_vector()

		if keep == False:
			image.clear()

		for j in range( len( object_properties_list ) ):
			if j != i:
				object_properties_list[ j ] = image.draw_object( object_properties_list[ j ] )
		image.add_noise( noise )

		hmax_feedback = HMAXFeedback.HMAXFeedback( image.getarray(), pooling, padding_noise = noise )

		hmax_feedback.feedback( map( lambda x: x > feedback_threshold and x or 0.0, numpy.array( attention_feature_vector ) ) )

		hmax_feedback_list.append( hmax_feedback )

	return hmax_feedback_list

def attend_template_and_plot( im, feedback_template, fig = None ):
	hmax_feedback = HMAXFeedback.HMAXFeedback( im.getarray() )

	hmax_feedback.feedback( feedback_template )

	plot_s2_mean_and_standard_deviation( hmax_feedback, f = fig );

def do_it( object_properties_list, keep = True, pooling = ( "Max", "Max" ), size = ( 64, 64 ), feedback_threshold = 0.0 ):
	"""Do the experiment. First a feeback template is generated. Then the an image with objects defined by object_properties_list is generated and fed through the hmax model. The feedback template is then used to perform the local consistency check.
	
	The object described by object_properties1 is the object used for the feedback template. The format for object properties is ( object, coords ) where object is a string for the object and coords are the coordinates, either teh string "random" or a 4-tupel ( x, y, width, height ).
	
	objects_properties_list is a list of object properties. Each entry in the list produces an object in the input image.
	
	If keep is true the object used for the feedback template is left in the input image, otherwise the image is cleared.
	
	pooling is the pooling of the hmax model. It is a tuple ( C1_pooling, C2_pooling ).
	
	Size defines the size of the input image."""
	hmax_feedback_list = do_attention( object_properties_list, keep, pooling, size, feedback_threshold )

	for i in range( len( hmax_feedback_list ) ):
		plot_s2_mean_and_standard_deviation( hmax_feedback_list[ i ], f = i + 1 )
	
	pylab.show()
	
	return hmax_feedback_list

def attend_and_plot_images( image1, image2, pooling = ( "Max", "Max" ), feedback_threshold = 0.0 ):
	attention_feature_vector = HMAXFeedback.HMAXFeedback( image1.getarray(), pooling ).feature_vector()

	hmax_feedback = HMAXFeedback.HMAXFeedback( image2.getarray(), pooling )

	hmax_feedback.feedback( map( lambda x: x > feedback_threshold and x or 0.0, numpy.array( attention_feature_vector ) ) )
	
	
	plot_s2_mean_and_standard_deviation( hmax_feedback, f = 1 )

	return hmax_feedback

def attend_and_plot( object_list, n = 0.2, size = ( 64, 64 ), feedback_threshold = 0.0 ):
	hl = do_attention( object_list, size = size, noise = n, feedback_threshold = feedback_threshold )
	pylab.figure( 1 )
	pylab.imshow( hl[ 0 ].input_image(), interpolation = interpol )

	for i in range( len( hl ) ):
		plot_s2_mean_and_standard_deviation( hl[ i ], f = i + 2 )
	
def plot_image_and_feature_vector( image, feature_vector, f, ( xmin, xmax, ymin, ymax ) = ( 0, 256, 0.3, 0.8 ) ):
	"""Plot an image and a feature vector on figure f."""
	pylab.figure( f )
	pylab.clf()
	pylab.subplot( 121 )
	pylab.imshow( image.getarray(), interpolation = interpol  )
	pylab.colorbar()	
	pylab.subplot( 122 )
	pylab.plot( range( xmin, xmax ), feature_vector )
	pylab.axes( [ xmin, xmax, ymin, ymax ] )
		
def plot_s2_max( hmax_feedback, max_min_level = 0.0, max_max_level = 1.0, min_min_level = 0.0, min_max_level = 1.0, f= 5 ):
	pylab.figure( f )
		
	max_min_level = numpy.max( numpy.array( [ numpy.min( hmax_feedback.lcc_s2_max( 0 ) ), numpy.min( hmax_feedback.lcc_s2_max( 1 ) ), numpy.min( hmax_feedback.lcc_s2_max( 2 ) ), numpy.min( hmax_feedback.lcc_s2_max( 3 ) ) ] ) )
	min_min_level = numpy.max( numpy.array( [ numpy.min( hmax_feedback.lcc_s2_min( 0 ) ), numpy.min( hmax_feedback.lcc_s2_min( 1 ) ), numpy.min( hmax_feedback.lcc_s2_min( 2 ) ), numpy.min( hmax_feedback.lcc_s2_min( 3 ) ) ] ) )
	
	max_max_level = numpy.max( numpy.array( [ numpy.max( hmax_feedback.lcc_s2_max( 0 ) ), numpy.max( hmax_feedback.lcc_s2_max( 1 ) ), numpy.max( hmax_feedback.lcc_s2_max( 2 ) ), numpy.max( hmax_feedback.lcc_s2_max( 3 ) ) ] ) )
	min_max_level = numpy.max( numpy.array( [ numpy.max( hmax_feedback.lcc_s2_min( 0 ) ), numpy.max( hmax_feedback.lcc_s2_min( 1 ) ), numpy.max( hmax_feedback.lcc_s2_min( 2 ) ), numpy.max( hmax_feedback.lcc_s2_min( 3 ) ) ] ) )

	
	pylab.clf()
	pylab.subplot( 241 )
	pylab.title( "LCC S2 min 0" )	
	pylab.imshow( hmax_feedback.lcc_s2_min( 0 ), vmin = min_min_level, vmax = min_max_level, interpolation = interpol  )
	pylab.subplot( 242 )
	pylab.title( "LCC S2 min 1" )
	pylab.imshow( hmax_feedback.lcc_s2_min( 1 ), vmin = min_min_level, vmax = min_max_level, interpolation = interpol  )
	pylab.subplot( 243 )
	pylab.title( "LCC S2 min 2" )
	pylab.imshow( hmax_feedback.lcc_s2_min( 2 ), vmin = min_min_level, vmax = min_max_level, interpolation = interpol  )
	pylab.subplot( 244 )
	pylab.title( "LCC S2 min 3" )
	pylab.imshow( hmax_feedback.lcc_s2_min( 3 ), vmin = min_min_level, vmax = min_max_level, interpolation = interpol  )
	pylab.colorbar()
	
	pylab.subplot( 245 )
	pylab.title( "LCC S2 max 0" )	
	pylab.imshow( hmax_feedback.lcc_s2_max( 0 ), vmin = max_min_level, vmax = max_max_level, interpolation = interpol  )
	pylab.subplot( 246 )
	pylab.title( "LCC S2 max 1" )
	pylab.imshow( hmax_feedback.lcc_s2_max( 1 ), vmin = max_min_level, vmax = max_max_level, interpolation = interpol  )
	pylab.subplot( 247 )
	pylab.title( "LCC S2 max 2" )
	pylab.imshow( hmax_feedback.lcc_s2_max( 2 ), vmin = max_min_level, vmax = max_max_level, interpolation = interpol  )
	pylab.subplot( 248 )
	pylab.title( "LCC S2 max 3" )
	pylab.imshow( hmax_feedback.lcc_s2_max( 3 ), vmin = max_min_level, vmax = max_max_level, interpolation = interpol  )
	pylab.colorbar()

def plot_s2_mean_and_standard_deviation( hmax_feedback, mean_min_level = 0.0, mean_max_level = 0.35, standard_deviation_min_level = 0.0, standard_deviation_max_level = 0.1, f = 4 ):
	"""Plot the mean and standard variance of the S2 feature maps in the local consitency map network."""
	pylab.figure( f )
	
	mean_min_level = numpy.max( numpy.array( [ numpy.min( hmax_feedback.lcc_s2_mean( 0 ) ), numpy.min( hmax_feedback.lcc_s2_mean( 1 ) ), numpy.min( hmax_feedback.lcc_s2_mean( 2 ) ), numpy.min( hmax_feedback.lcc_s2_mean( 3 ) ) ] ) )
	standard_deviation_min_level = numpy.min( numpy.array( [ numpy.min( hmax_feedback.lcc_s2_standard_deviation( 0 ) ), numpy.min( hmax_feedback.lcc_s2_standard_deviation( 1 ) ), numpy.min( hmax_feedback.lcc_s2_standard_deviation( 2 ) ), numpy.min( hmax_feedback.lcc_s2_standard_deviation( 3 ) ) ] ) )

	mean_max_level = numpy.max( numpy.array( [ numpy.max( hmax_feedback.lcc_s2_mean( 0 ) ), numpy.max( hmax_feedback.lcc_s2_mean( 1 ) ), numpy.max( hmax_feedback.lcc_s2_mean( 2 ) ), numpy.max( hmax_feedback.lcc_s2_mean( 3 ) ) ] ) )
	standard_deviation_max_level = numpy.max( numpy.array( [ numpy.max( hmax_feedback.lcc_s2_standard_deviation( 0 ) ), numpy.max( hmax_feedback.lcc_s2_standard_deviation( 1 ) ), numpy.max( hmax_feedback.lcc_s2_standard_deviation( 2 ) ), numpy.max( hmax_feedback.lcc_s2_standard_deviation( 3 ) ) ] ) )

	print "mean values: ", mean_min_level, mean_max_level
	print "standard deviation values: ", standard_deviation_min_level, standard_deviation_max_level
	
	pylab.clf()
	pylab.subplot( 241 )
#	pylab.title( "LCC S2 mean 0" )	
	pylab.imshow( hmax_feedback.lcc_s2_mean( 0 ), vmin = mean_min_level, vmax = mean_max_level, interpolation = interpol  )
	pylab.subplot( 242 )
#	pylab.title( "LCC S2 mean 1" )
	pylab.imshow( hmax_feedback.lcc_s2_mean( 1 ), vmin = mean_min_level, vmax = mean_max_level, interpolation = interpol  )
	pylab.subplot( 243 )
#	pylab.title( "LCC S2 mean 2" )
	pylab.imshow( hmax_feedback.lcc_s2_mean( 2 ), vmin = mean_min_level, vmax = mean_max_level, interpolation = interpol  )
	pylab.subplot( 244 )
#	pylab.title( "LCC S2 mean 3" )
	pylab.imshow( hmax_feedback.lcc_s2_mean( 3 ), vmin = mean_min_level, vmax = mean_max_level, interpolation = interpol  )
	pylab.colorbar()
	pylab.subplot( 245 )
#	pylab.title( "LCC S2 variance 0" )
	pylab.imshow( hmax_feedback.lcc_s2_standard_deviation( 0 ), vmin = standard_deviation_min_level, vmax = standard_deviation_max_level, interpolation = interpol  )
	pylab.subplot( 246 )
#	pylab.title( "LCC S2 variance 1" )
	pylab.imshow( hmax_feedback.lcc_s2_standard_deviation( 1 ), vmin = standard_deviation_min_level, vmax = standard_deviation_max_level, interpolation = interpol  )
	pylab.subplot( 247 )
#	pylab.title( "LCC S2 variance 2" )
	pylab.imshow( hmax_feedback.lcc_s2_standard_deviation( 2 ), vmin = standard_deviation_min_level, vmax = standard_deviation_max_level, interpolation = interpol  )
	pylab.subplot( 248 )
#	pylab.title( "LCC S2 variance 3" )
	pylab.imshow( hmax_feedback.lcc_s2_standard_deviation( 3 ), vmin = standard_deviation_min_level, vmax = standard_deviation_max_level, interpolation = interpol  )
	pylab.colorbar()

def data_string( file_name , ( object_description1, object_description2 ), size ):
		width, height = size
		( object1, ( ( ( x1, y1 ), ( h1, w1 ) ), ( pen_width1, color1, opacity1 ) ) ) = object_description1
		( object2, ( ( ( x2, y2 ), ( h2, w2 ) ), ( pen_width2, color2, opacity2 ) ) ) = object_description2
		r = file_name + "	"
		r = r + object1 + "	"
		r = r + ( "%f" % ( float( x1 ) / float( width ) ) ) + "	"
		r = r + ( "%f" % ( float( y1 ) / float( height ) ) ) + "	"
		r = r + ( "%f" % ( float( w1 ) / float( width ) ) ) + "	"
		r = r + ( "%f" % ( float( h1 ) / float( height ) ) ) + "	"
		r = r + ( "%d" % pen_width1 ) + "	" 
		r = r + color1 + "	" 
		r = r + ( "%d" % opacity1 ) + "	"
		r = r + object2 + "	"
		r = r + ( "%f" % ( float( x2 ) / float( width ) ) ) + "	"
		r = r + ( "%f" % ( float( y2 ) / float( height ) ) ) + "	"
		r = r + ( "%f" % ( float( w2 ) / float( width ) ) ) + "	"
		r = r + ( "%f" % ( float( h2 ) / float( height ) ) ) + "	"
		r = r + ( "%d" % pen_width2 ) + "	"
		r = r + color2 + "	"
		r = r + ( "%d" % opacity2 ) + "	"
	
		return r

def write_rawinput( f, image, ( file_name, ( object_description1, object_description2 ), size ) ):
	f.write( data_string( file_name, ( object_description1, object_description2 ), size ) )
	for j in numpy.array( image.getdata() ):
		f.write( ( "%f" % ( j / 255.0 ) ) + """	""" )

	f.write( "\n" )


def write_lcc_s2( f, hmax_feedback, ( file_name, ( object_description1, object_description2 ), size ) ):
	f.write( data_string( file_name, ( object_description1, object_description2 ), size ) )
		
	for x in range( 0, 4 ):
		for j in hmax_feedback.lcc_s2_mean( x ).reshape( hmax_feedback.lcc_s2_mean( x ).shape[ 0 ] * hmax_feedback.lcc_s2_mean( x ).shape[ 1 ] ):
			f.write( ( "%f" % j ) + """	""" )

	for x in range( 0, 4 ):
		for j in hmax_feedback.lcc_s2_standard_deviation( x ).reshape( hmax_feedback.lcc_s2_standard_deviation( x ).shape[ 0 ] * hmax_feedback.lcc_s2_standard_deviation( x ).shape[ 1 ] ):
			f.write( ( "%f" % j ) + """	""" )

	f.write( "\n" )

def vertical_vs_horizontal_bar( o1, o2, pen_width ):
	hb = ( o1, ( ( ( 5, 5 ), ( 25, 25 ) ), ( pen_width, "white", 255 ) ) )
	vb = ( o2, ( ( ( 35, 35 ), ( 25, 25 ) ), ( pen_width, "white", 255 ) ) )
	 
	hmax = do_attention( [ hb, vb ] )
	pylab.figure( 1 )
	pylab.imshow( hmax[ 0 ].input_image() )
	
	plot_s2_mean_and_standard_deviation( hmax[ 0 ], f = 2 )
	plot_s2_max( hmax[ 0 ], f = 3 )
	plot_s2_mean_and_standard_deviation( hmax[ 1 ], f = 4 )
	plot_s2_max( hmax[ 1 ], f = 5 )

def orange_header_feature_vector( nr_features, object_classes ):
	"""Returns the orange header of a feature vector of length nr_features with len( object_classes ) object classes."""
	
	s = ""
	for i in range( nr_features ):
		s += """f%d	""" % i
		
	for i in object_classes:
		s += i + """	"""

	s += "\n"
	for i in range( nr_features ):
		s += """continuous	"""
	
	for i in object_classes:
		s += """discrete	"""
		
	s += "\n"
	
	return s
