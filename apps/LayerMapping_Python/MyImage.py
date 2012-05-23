"""Image with routines to draw objects.
	author: Johannes Drever"""

import Image
import aggdraw

import random

import numpy

def open( file_name ):
	im = MyImage( ( 1, 1 ) )
	im.load( file_name )

	return im

class MyImage:
	"""Image class."""
	def __init__( self, size, color = "black", no_duplicate_objects = True, margin = 0, max_pen_width = 5, occlusion = False ):
		self.color = color
		self.im = Image.new( "L", size, color )
		
		self.draw = aggdraw.Draw( self.im )
		
		self.objects_drawn = []
		self.no_duplicate_objects = no_duplicate_objects

		self.margin = margin
		
		self.max_pen_width = max_pen_width

		self.position_list = []

	def save( self, f ):
		"""Save the image in PNG to file f."""
		return self.im.save( f +  ".png", "PNG" )

	def load( self, f ):
		"""Load the image from file f. The image is converted into grayscale."""
		self.im = Image.open( f ).convert( "L" )
		
	def getdata( self ):
		"""Get the raw image data"""
		return self.im.getdata()
	
	def getarray( self ):
		"""Get data in form of a numpy.array. pylab can use this for plotting."""
		#( height, width ) = self.im.size
		#size = ( width, height )
		return numpy.array( self.im.getdata() ).reshape( self.im.size )

	def clear( self ):
		"""Clear the image."""
		self.__init__( self.im.size, self.color )
		
	def rectangle( self, ( x, y ), ( w, h ), pen ):
		"""Draw a rectangle at position (x,y) with width w and height h.pen is an aggdraw pen."""
		self.draw.rectangle( ( x, y, x + w, y + h ), pen )
		self.draw.flush()
	
	def diamond( self, ( x, y ), ( w, h ), pen ):
		"""Draw a diamond at position (x,y) with width w and height h.
pen is an aggdraw pen."""
		self.draw.polygon( ( x + w / 2, y, x + w, y + h / 2, x + w / 2, y + h, x, y + h / 2 ), pen )
		self.draw.flush()
	
	def triangle( self, ( x, y ), ( w, h ), pen ):
		"""Draw a triangle at position (x,y) with width w and height h.
pen is an aggdraw pen."""

		self.draw.polygon( ( x, y, x + w, y, x + ( w / 2 ), y + h ), pen )
		self.draw.flush()		
						
	def circle( self, ( x, y ), ( w, h ), pen ):
		"""Draw a circle at position (x,y) with width w and height h.
pen is an aggdraw pen."""

		self.draw.ellipse( ( x, y, x + h, y + w ), pen )
		self.draw.flush()
				
	def cross( self, ( x, y ), ( w, h ), pen ):
		"""Draw a cross at position (x,y) with width w and height h.
pen is an aggdraw pen."""

		self.draw.line( ( x + ( w / 2 ), y, x + ( w / 2 ), y + w ), pen )
		self.draw.line( ( x, y + ( h / 2 ), x + w, y + ( w / 2 ) ), pen )
		self.draw.flush()
	
	def object1( self, ( x, y ), ( w, h ), pen ):
		"""Draw an object1 at position (x,y) with width w and height h.
pen is an aggdraw pen."""


		self.draw.line( ( x + ( w * 1 ), y + ( h * 0.2 ), \
			x + ( w * 0.4 ), y + ( h * 0 ), \
			x + ( w * 0 ), y + ( h * 1 ), \
			x + ( w * 0.4 ), y + ( h * 0.6 ), \
			x + ( w * 0.8 ), y + ( h * 0.8 ), \
			x + ( w * 0.5 ), y + ( h * 0.1 ) ), pen )
		self.draw.flush()

	def object2( self, ( x, y ), ( w, h ), pen ):
		"""Draw an object2 at position (x,y) with width w and height h.
pen is an aggdraw pen."""

		self.draw.line( ( x + ( w * 0.5 ), y + ( h * 0 ), \
			x + ( w * 0.5 ), y + ( h * 0.7 ), \
			x + ( w * 0.0 ), y + ( h * 0.5 ), \
			x + ( w * 0.8 ), y + ( h * 0.2 ), \
			x + ( w * 1 ), y + ( h * 0.8 ), \
			x + ( w * 0.5 ), y + ( h * 0.4 ) ), pen )
		self.draw.flush()

	def object3( self, ( x, y ), ( w, h ), pen ):
		"""Draw an object3 at position (x,y) with width w and height h.
pen is an aggdraw pen."""

		self.draw.line( ( x + ( w * 0.5 ), y + ( h * 0.0 ), \
			x + ( w * 0.8 ), y + ( h * 0.2 ), \
			x + ( w * 1 ), y + ( h * 0.6 ), \
			x + ( w * 0.4 ), y + ( h * 0.6 ), \
			x + ( w * 0.0 ), y + ( h * 0.8 ), \
			x + ( w * 0.0 ), y + ( h * 0.1 ) ), pen )
		self.draw.flush()

	def object4( self, ( x, y ), ( w, h ), pen ):
		"""Draw an object4 at position (x,y) with width w and height h.
pen is an aggdraw pen."""
		self.draw.line( ( x + ( w * 0.5 ), y + ( h * 0.7 ), \
			x + ( w * 0.0 ), y + ( h * 0.0 ), \
			x + ( w * 0.5 ), y + ( h * 0.0 ), \
			x + ( w * 0.9 ), y + ( h * 0.6 ), \
			x + ( w * 0.9 ), y + ( h * 0.1 ), \
			x + ( w * 0.6 ), y + ( h * 0.0 ) ), pen )
		self.draw.flush()

	def object5( self, ( x, y ), ( w, h ), pen ):
		"""Draw an object5 at position (x,y) with width w and height h.
pen is an aggdraw pen."""

		self.draw.line( ( x + ( w * 1 ), y + ( h * 0.6 ), \
			x + ( w * 0.4 ), y + ( h * 0.6 ), \
			x + ( w * 0.6 ), y + ( h * 1 ), \
			x + ( w * 0.2 ), y + ( h * 0.5 ), \
			x + ( w * 0.8 ), y + ( h * 0.1 ), \
			x + ( w * 0.2 ), y + ( h * 0.9 ) ), pen )
		self.draw.flush()

	def horizontal_bar( self, ( x, y ), ( w, h ), pen ):
		"""Draw a horizontal_bar at position (x,y) with width w and height h.
pen is an aggdraw pen."""

		self.draw.line( ( x, y + ( h / 2 ), x + w, y + ( w / 2 ) ), pen )
		self.draw.flush()
			
	def vertical_bar( self, ( x, y ), ( w, h ), pen ):
		"""Draw a vertical_bar at position (x,y) with width w and height h. pen is an aggdraw pen."""

		self.draw.line( ( x + ( w / 2 ), y, x + ( w / 2 ), y + h ), pen )
		self.draw.flush()
				
	def diagonal1( self, ( x, y ), ( w, h ), pen ):
		"""Draw a diagonal1 at position (x,y) with width w and height h.
pen is an aggdraw pen."""

		self.draw.line( ( x, y, x + w, y + h ), pen )
		self.draw.flush()
				
	def diagonal2( self, ( x, y ), ( w, h ), pen ):
		"""Draw a diagonale2 at position (x,y) with width w and height h. pen is an aggdraw pen."""

		self.draw.line( ( x + w, y, x, y + h ), pen )
		self.draw.flush()
	
	def star( self, ( x, y ), ( w, h ), pen ):
		"""Draw a star at position (x,y) with width w and height h.
pen is an aggdraw pen."""

		xx = x + ( w / 2 )
		yy = y + ( h / 2 )
		
		x1 = xx + w * numpy.cos( 0 ) / 2
		y1 = yy + h * numpy.sin( 0 ) / 2
		x2 = xx + w * numpy.cos( 2.0 * 2.0 * numpy.pi / 5.0 ) / 2
		y2 = yy + h * numpy.sin( 2.0 * 2.0 * numpy.pi / 5.0 ) / 2
		x3 = xx + w * numpy.cos( 4.0 * 2.0 * numpy.pi / 5.0 ) / 2
		y3 = yy + h * numpy.sin( 4.0 * 2.0 * numpy.pi / 5.0 ) / 2
		x4 = xx + w * numpy.cos( 1.0 * 2.0 * numpy.pi / 5.0 ) / 2
		y4 = yy + h * numpy.sin( 1.0 * 2.0 * numpy.pi / 5.0 ) / 2
		x5 = xx + w * numpy.cos( 3.0 * 2.0 * numpy.pi / 5.0 ) / 2
		y5 = yy + h * numpy.sin( 3.0 * 2.0 * numpy.pi / 5.0 ) / 2
		
		self.draw.polygon( ( x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 ), pen )
		self.draw.flush()

	def add_noise( self, level ):
		"""Add noise to the image."""
		max_level = 255
		lev = level * max_level
		l = list( self.im.getdata() )
		for i in range( 0, len( l ) ):
			l[ i ] = min( l[ i ] + random.uniform( 0, lev ), max_level )
		self.im.putdata( l )
		
	def objects( self ):
		"""The objects that can be drawn to the image by draw_object."""
		return [ "horizontal_bar", "vertical_bar", "rectangle", "diamond", "triangle", "circle", "cross", "diagonal1", "diagonal2", "star", "object1" ]
		
	def draw_object( self, ( obj, object_properties ) = ( "random", "random" ), possible_objects = "all" ):
		"""Draw an object to the image. obj is the object to be drawn and can be either random or an object in the list possible_objects. If possible_objects is "all" the list returned by objects() is used. object_properties can be either random or in the form ( ( ( x, y ), ( w, h ), ), pen ), where pen is defined according to aggdraw. """
		if object_properties == "random":
			( coords, ( pw, c, opacity ) ) = ( "random", ( "random", "white", 255 ) )
		else:
			( coords, ( pw, c, opacity ) ) = object_properties
		width, height = self.im.size

		if possible_objects == "all":
			possible_objects = self.objects()

		if obj == "random":
			if self.no_duplicate_objects == True:
				objects_left = filter( lambda x: x not in self.objects_drawn, possible_objects )
				if( objects_left == [] ):
					raise "Sorry, no objects left."
				obj = random.sample( objects_left, 1 )[ 0 ]
			else:
				obj = random.sample( possible_objects, 1 )[ 0 ]				
			

		self.objects_drawn = self.objects_drawn + [ obj ]

		if pw == "random":
			pwidth = random.randint( 1, self.max_pen_width )
		elif True:
			pwidth = pw

		if coords == "random":
			done = False
			while( not done ):
				x = random.randint( self.margin + pwidth, width - self.margin - pwidth )
				y = random.randint( self.margin + pwidth, height - self.margin - pwidth )
				done = True
				try:
					s = random.randint( 3, width - ( max( x, y ) + self.margin + pwidth * 2 ) )
				except:
					done = False
		elif True:
			( ( x, y ), ( s, s ) ) = coords
			#TODO define bounding constraints and raise exceptions if they are violated!
			# if x < margin | x + s > margin:
		
		o = 255
		
		pen = aggdraw.Pen( color = c, width = pwidth, opacity = opacity )
		
		if obj == "rectangle":
			self.rectangle( ( x, y ), ( s, s ), pen )
		elif obj == "diamond":
			self.diamond( ( x, y ), ( s, s ), pen )
		elif obj == "triangle":
			self.triangle( ( x, y ), ( s, s ), pen )
		elif obj == "circle":
			self.circle( ( x, y ), ( s, s ), pen )
		elif obj == "cross":
			self.cross( ( x, y ), ( s, s ), pen )
		elif obj == "vertical_bar":
			self.vertical_bar( ( x, y ), ( s, s ), pen )
		elif obj == "horizontal_bar":
			self.horizontal_bar( ( x, y ), ( s, s ), pen )
		elif obj == "diagonal1":
			self.diagonal1( ( x, y ), ( s, s ), pen )
		elif obj == "diagonal2":
			self.diagonal2( ( x, y ), ( s, s ), pen )
		elif obj == "star":
			self.star( ( x, y ), ( s, s ), pen )
		elif obj == "object1":
			self.object1( ( x, y ), ( s, s ), pen )
		elif obj == "object2":
			self.object2( ( x, y ), ( s, s ), pen )
		elif obj == "object3":
			self.object3( ( x, y ), ( s, s ), pen )
		elif obj == "object4":
			self.object4( ( x, y ), ( s, s ), pen )
		elif obj == "object5":
			self.object5( ( x, y ), ( s, s ), pen )
		elif True:
			raise NameError, "Unkown object: " + obj
		
		return ( obj, ( ( ( x, y ), ( s, s ) ), ( pwidth, c, o ) ) )
