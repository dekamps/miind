import matplotlib
import matplotlib.pyplot as plt
import sys
import ast
import xml.etree.ElementTree as ET


def write_fid(fid_fname, quadlist):
    with open(fid_fname, 'w') as f:
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
    f = open(fn)
    lines = f.readlines()
    x = [float(item.split()[0]) for item in lines]
    y = [float(item.split()[1]) for item in lines]
    return x, y


def plot_lost(fn):
    ax = plt.gcf().add_subplot(111)
    x, y = read_file(fn)
    plt.plot(x, y, '.')
    return ax


def add_fiducial(ax, point_list):
    from matplotlib.path import Path
    verts = [(x[0], x[1]) for x in point_list]
    verts.append((point_list[0][0], point_list[0][1]))

    codes = [Path.MOVETO]
    for x in range(1, len(point_list)):
        codes.append(Path.LINETO)
    codes.append(Path.CLOSEPOLY)

    path  = Path(verts, codes)
    patch = matplotlib.patches.PathPatch(path, color='orange')
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

        points = list(zip(vs,ws))
        patches.append(points)
    return patches


def extract_base(fn):
    return fn.replace('.lost', '').split('_')[-6]


def zoom_fun(event, ax, base_scale=1.5):
    # get the current x and y limits
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
    cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
    xdata = event.xdata # get event x location
    ydata = event.ydata # get event y location
    if event.button == 'up':
        # deal with zoom in
        scale_factor = 1/base_scale
    elif event.button == 'down':
        # deal with zoom out
        scale_factor = base_scale
    else:
        # deal with something that should never happen
        scale_factor = 1
        print(event.button)
    # set new limits
    ax.set_xlim([xdata - cur_xrange*scale_factor,
                 xdata + cur_xrange*scale_factor])
    ax.set_ylim([ydata - cur_yrange*scale_factor,
                 ydata + cur_yrange*scale_factor])
    plt.draw() # force re-draw


def onclick(event, ax, fid_fname, curr_points, quads):

    toolbar = plt.get_current_fig_manager().toolbar
    if toolbar.mode!='':
        return

    if event.dblclick:
        write_fid(fid_fname, quads)
        plt.close()

    inv = ax.transData.inverted()
    coords = inv.transform((event.x, event.y))
    if len(curr_points) < 4:
        curr_points.append(coords)
        plt.plot(coords[0], coords[1], 'r+')
        plt.draw()
    else:
        del curr_points[:]
        curr_points.append(coords)
        plt.plot(coords[0], coords[1], 'r+')
        plt.draw()
    if len(curr_points) == 4:
        patch = [point for point in curr_points]
        quads.append(patch)
        add_fiducial(ax, patch)


def onkey(event, ax, fid_fname, quads):
    if event.key == 'd':
        print('Deleting previous Fiducial.')
        del quads[-1]
        write_fid(fid_fname, quads)
        ax.patches[-1].remove()
        plt.draw()
    elif event.key == 'c':
        print('Clearing canvas and redrawing fiducials.')
        plt.cla()
        for patch in read_fiducial(fid_fname):
            add_fiducial(ax, patch)
        plt.draw()
    elif event.key == 'return':
        write_fid(fid_fname, quads)
        plt.close()
    else:
        print('Key "{}" not recognized.'.format(event.key))
        return
