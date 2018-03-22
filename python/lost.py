import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import ast
import xml.etree.ElementTree as ET
from matplotlib.path import Path


curr_points = []
quads = []

def write_fid(bn, quadlist):
    with open(bn + '.fid','w') as f:
        f.write('<Fiducial>\n')
        for quad in quadlist:

            vs = [ point[0] for point in quad ]
            ws = [ point[1] for point in quad ]

            f.write('<Quadrilateral type= \"Contain\">')
            f.write('<vline>')
            for v in vs:
                f.write(str(v) + ' ')
            f.write('</vline>')
            f.write('<wline>')
            for w in ws:
                f.write(str(w) + ' ')
            f.write('</wline>')
            f.write('</Quadrilateral>\n')
        f.write('</Fiducial>\n')

def read_file(fn):
    f=open(fn)
    lines=f.readlines()

    x=[ float(item.split()[0]) for item in lines ]
    y=[ float(item.split()[1]) for item in lines ]

    return x, y

def plot_lost(fn):

    ax = fig.add_subplot(111)
    x,y = read_file(fn)
    plt.plot(x,y,'.')
    return ax


def add_fiducial(ax,point_list):
    verts = [ (x[0], x[1]) for x in point_list ]
    verts.append( (point_list[0][0], point_list[0][1]) )

    codes = [Path.MOVETO]
    for x in range(1,len(point_list)):
        codes.append(Path.LINETO)
    codes.append(Path.CLOSEPOLY)

    path  = Path(verts, codes)
    patch = patches.PathPatch(path, color='orange',lw=2)
    ax.add_patch(patch)



def read_fiducial(fn):
    patches = []

    tree = ET.parse(fn)
    root = tree.getroot()
    for child in root:
        if child.tag != "Quadrilateral":
            raise ValueError
        babys = [baby for baby in child]

        vs = [ float(x) for x in  babys[0].text.split() ]
        ws = [ float(x) for x in  babys[1].text.split() ]

        points = zip(vs,ws)
        patches.append(points)
        quads.append(points)
    return patches

def extract_base(fn):
    return fn.split('.')[0].split('_')[0]

def main(args):
    if len(args) != 2:
        print 'Usage: \' python lost.py <filename>.lost \' '
        raise SystemExit()

    fig = plt.figure()
    ax = plot_lost(args[1])
    bn = extract_base(args[1])
    l=read_fiducial(bn + '.fid')
    for patch in l:
        add_fiducial(ax,patch)

    def onclick(event):
        global bn, cid
        if event.dblclick:
            write_fid(bn,quads)
            raise SystemExit

        inv = ax.transData.inverted()
        coords = inv.transform( (event.x, event.y) )
        if len(curr_points) < 4:
            curr_points.append(coords)
            plt.plot(coords[0],coords[1],'r+')
            plt.draw()
        else:
            patch = [ point for point in curr_points]
            quads.append(patch)
            add_fiducial(ax,patch)

            del curr_points[:]
            curr_points.append(coords)
            plt.plot(coords[0],coords[1],'r+')
            plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)


    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
