import numpy as np

def isonsameside(line,x,y):
    '''Determines whether the points x, y are on the same side of a line.
    line cosist of a begin point line[0] and and end point line[1]. Each point
    has an x-coordinate [0] and a y-coordinate [1]. If at least one of the points
    is on the line, both points are considered to be  on the same side of the line.'''

    bmina = line[1] - line[0]


    xmina = x - line[0]
    ymina = y - line[0]

    # z components of cross product
    z_cross_x = bmina[0]*xmina[1] - bmina[1]*xmina[0]
    z_cross_y = bmina[0]*ymina[1] - bmina[1]*ymina[0]

    if z_cross_x*z_cross_y >= 0:
        return True
    else:
        return False



def BaryCentric(p, a, b, c):
    '''From http://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates.
    a,b,c must numpy arrays'''
    
    v0 = b - a
    v1 = c - a
    v2 = p - a;
    d00 = np.dot(v0, v0);
    d01 = np.dot(v0, v1);
    d11 = np.dot(v1, v1);
    d20 = np.dot(v2, v0);
    d21 = np.dot(v2, v1);
    denom = d00 * d11 - d01 * d01;
    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0 - v - w;
    return u, v, w


def isinsidetriangle(triangle, point):
    ''' triangle is  ia list of 3x2 coordinates (or a compatible type). Point is a 
    list of two coordinates.'''

    a = triangle[0]
    b = triangle[1]
    c = triangle[2]
    u, v, w = BaryCentric(point, a, b, c)

    # points on the triangle are considered to be inside! insidequadrilateral depends on that
    if u > 1 or u < 0 or v > 1 or v < 0 or w > 1 or w < 0:
        return False
    else:
        return True
    

def splitquadrilateral(quadrilateral):
    ''' Split the quadrilateral into two triangles, deal correctly with concave quadrilaterals.'''
    a=quadrilateral[0]
    b=quadrilateral[1]
    c=quadrilateral[2]
    d=quadrilateral[3]
    
    # Consider the the diagonal a,c if d and b are on the same side, we have to use the other diagonal to
    # bisect the quad.

    line = np.array([a,c])

    if isonsameside(line, b,d):

        triangle1 = [a, b, d]
        triangle2 = [b, d, c]

        return triangle1, triangle2
    else:
        # test triangles a c b and a c d

        triangle1 = [a, c, b]
        triangle2 = [a, c, d]
        return triangle1, triangle2


def isinsidequadrilateral(quadrilateral, point):
    ''' The quadrilateral is assumed not to be self intersecting. The routine splits
    the quadrilateral into two triangles. It then tests wether it is in those triangles or not.
    It correctly deals with concave quadrilaterals. It returns a 3-tuple of a bool and the two triangles
    in which it has split them.'''

    triangle1, triangle2 = splitquadrilateral(quadrilateral)
    if isinsidetriangle(triangle1,point) or isinsidetriangle(triangle2,point):
        return True
    else:
        return False
    


def areaoftriangle(triangle):
    '''Assumes non-degeneracy. Will throw and exception on the sqrt if this not true.'''

    x = triangle[0]
    y = triangle[1]
    z = triangle[2]

    a = np.sqrt(np.dot(y-x,y-x))
    b = np.sqrt(np.dot(z-y,z-y))
    c = np.sqrt(np.dot(x-z,x-z))
 
    s = (a+b+c) / 2.0
    w = s*(s-a)*(s-b)*(s-c)
    if w < 0:
        print'Degenerate triangle'
        print x
        print y
        print z
        raise ValueError
    else:
        area = np.sqrt(w)
    return area

def areaofquadrilateral(quadrilateral):
    ''' Calculates area of non intersecting quadrilateral.'''
    point = np.array([0.,0.])
    triangle1, triangle2 = splitquadrilateral(quadrilateral)
    return areaoftriangle(triangle1) + areaoftriangle(triangle2)

def generate_random_triangle_points(triangle, N):
    '''Create N uniformly distributed points inside a triangle.'''
    points=[]
    tot = 0 
    while tot < N:
        u = np.random.uniform(0.,1.)
        v = np.random.uniform(0.,1.)
        if u+v <= 1:
            tot += 1
            points.append(triangle[0] + u*(triangle[1] - triangle[0]) + v*(triangle[2] - triangle[0]))
    return np.array(points)


def generate_random_quadrilateral_points(quadrilateral, N):
    '''Create N uniformly distributed points inside a quadrilateral.'''

    triangle1 , triangle2 = splitquadrilateral(quadrilateral)
    area1 = areaoftriangle(triangle1)
    area2 = areaoftriangle(triangle2)
    p = area1/(area1 + area2)
    
    points = []
    for i in range(N):
        x=np.random.uniform(0.,1.)
        if x < p:
            points.append(generate_random_triangle_points(triangle1,1)[0])
        else:
            points.append(generate_random_triangle_points(triangle2,1)[0])

    return points


def distance_to_line(line, point):
    ''' The line is a list of two points. Returns the distance of the point to the line.'''
    p1 = line[0]
    p2 = line[1]

    num   = np.fabs( (p2[1] - p1[1])*point[0] - (p2[0] - p1[0])*point[1] + p2[0]*p1[1] - p2[1]*p1[0])
    denum = np.sqrt( (p2[0] - p1[0])*(p2[0] - p1[0]) + (p2[1] -p1[1])*(p2[1] -p1[1]) )

    if denum == 0:
        raise ValueError
    return num/denum


def distance_to_line_segment(line_segment, point):
    ''' The line segment is a list of two points. Returns the distance to  the line segment.'''
    p1 = np.array(line_segment[0],dtype = float)
    p2 = np.array(line_segment[1],dtype = float)
    p  = np.array(point, dtype = float)
    l2 = np.dot(p2 - p1, p2 - p1)
    if l2 == 0:
        return np.sqrt( np.dot(p - p1, p - p1))
    t = np.dot( p - p1, p2 - p1)/l2

    if t < 0 :
        return np.sqrt( np.dot( p - p1, p - p1) )
    if t > 1 :
        return np.sqrt( np.dot( p - p2, p - p2) )

    lp = p1 + t*(p2 - p1)

    return np.sqrt( np.dot( p - lp, p - lp) )
    

def distance_to_quadrilateral(quadrilateral, point):
    ''' Returns the shortest distance to the boundary of the quadrilateral.'''
    if len(quadrilateral) != 4:
        raise ValueError

    distances = [  distance_to_line_segment([quadrilateral[i], quadrilateral[(i+1)%4]], point) for i in range(len(quadrilateral)) ]
    d = np.min(distances)
    return d
        


# adapted from: http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/1201356#1201356

def get_line_intersection(p0t, p1t, p2t, p3t):

    ''' Calculates the intersection point between line segments p0,p1 and p2,p3. The points are converted to np.arrays of type float64, so no integer calculations
    are performed when the points are presented as integers. Returns a three tuple ret. If there are no
    interesections ret[0] is 0. In that case ret[1] and ret[2] should not be used. If ret[1] is equal to 1, then there is a
    collision. Its x coordinate is ret[1], and its y coordinate ret[2].'''
    
    p0 = np.array(p0t,dtype=np.float64)
    p1 = np.array(p1t,dtype=np.float64)
    p2 = np.array(p2t,dtype=np.float64)
    p3 = np.array(p3t,dtype=np.float64)

    s10_x = p1[0] - p0[0]
    s10_y = p1[1] - p0[1]
    s32_x = p3[0] - p2[0]
    s32_y = p3[1] - p2[1]

    denom = s10_x * s32_y - s32_x * s10_y
    if (denom == 0):
        return 0,0,0 # Collinear

    denomPositive = denom > 0;

    s02_x = p0[0] - p2[0];
    s02_y = p0[1] - p2[1];
    s_numer = s10_x * s02_y - s10_y * s02_x;
    if ((s_numer < 0) == denomPositive):
        return 0,0,0 # No collision

    t_numer = s32_x * s02_y - s32_y * s02_x;
    if ((t_numer < 0) == denomPositive):
        return 0,0,0 # No collision

    if (((s_numer > denom) == denomPositive) or ((t_numer > denom) == denomPositive)):
        return 0,0,0 # No collision
    # Collision detected
    t = t_numer / denom;

    i_x = p0[0] + (t * s10_x);
    i_y = p0[1] + (t * s10_y);

    return 1,i_x,i_y;


