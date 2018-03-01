import click
from miindio import MiindIO
import matplotlib
import matplotlib.pyplot as plt
import copy


@click.group()
def cli():
    pass


@cli.command("submit", short_help='Generate a MIIND executable based on a ' +
             '"xml" parameter file.')
@click.argument("xml_path", type=click.Path(exists=True))
@click.option("--directory", "-d", type=click.Path(exists=True))
@click.option("--run", "-r", is_flag=True)
def generate_(xml_path, **kwargs):
    io = MiindIO(xml_path, kwargs['directory'])
    io.submit()
    if kwargs['run']:
        io.run()


@cli.command("run", short_help='Run a MIIND executable based on a ' +
             '"xml" parameter file.')
@click.argument("xml_path", type=click.Path(exists=True))
@click.option("--directory", "-d", type=click.Path(exists=True))
@click.option("--generate", "-g", is_flag=True)
def run_(xml_path, **kwargs):
    io = MiindIO(xml_path, kwargs['directory'])
    if kwargs['generate']:
        io.submit()
    io.run()


@cli.command("plot-marginal-density",
             short_help='Plot marginal density of a model')
@click.argument("xml_path", type=click.Path(exists=True))
@click.argument("model_name", type=click.STRING)
@click.option("--directory", "-d", type=click.Path(exists=True))
@click.option("--n_bins_w", "-w", default=100, type=click.INT)
@click.option("--n_bins_v", "-v", default=100, type=click.INT)
def plot_marginal_density_(xml_path, model_name, **kwargs):
    io = MiindIO(xml_path, kwargs['directory'])
    marginal = io.marginal[model_name]
    marginal.vn = kwargs['n_bins_v']
    marginal.wn = kwargs['n_bins_w']
    marginal.plot()


@cli.command("plot-density",
             short_help='Plot 2D density of a model')
@click.argument("xml_path", type=click.Path(exists=True))
@click.argument("model_name", type=click.STRING)
@click.option("--directory", "-d", type=click.Path(exists=True))
@click.option("--timestep", "-t", type=click.INT)
def plot_density_(xml_path, model_name, **kwargs):
    io = MiindIO(xml_path, kwargs['directory'])
    density = io.density[model_name]
    for fname in density.fnames[::kwargs['timestep']]:
        density.plot_density(fname, save=True)


@cli.command("lost",
             short_help='Plot .lost file')
@click.argument("lost_path", type=click.Path(exists=True))
def plot_lost_(lost_path, **kwargs):
    from miindio.lost_tools import (add_fiducial, extract_base,
                                    plot_lost, read_fiducial,
                                    onclick, zoom_fun, onkey)
    backend = matplotlib.get_backend().lower()
    if backend not in ['qt4agg']:
        print('Warning: backend not recognized as working with "lost.py", ' +
              'if you do not encounter any issues with your current backend ' +
              '{}, please add it to this list.'.format(backend))
    curr_points = []
    fig = plt.figure()
    ax = plot_lost(lost_path)
    fid_fname = extract_base(lost_path) + '.fid'
    patches = read_fiducial(fid_fname)
    quads = copy.deepcopy(patches)
    for patch in patches:
        add_fiducial(ax, patch)

    fig.canvas.mpl_connect('button_press_event',
                           lambda event: onclick(event, ax, fid_fname,
                                                 curr_points, quads))
    fig.canvas.mpl_connect('scroll_event', lambda event: zoom_fun(event, ax))
    fig.canvas.mpl_connect('key_press_event',
                           lambda event: onkey(event, ax, fid_fname, quads))
    plt.show()


def main():
    cli()

if __name__ == "__main__":
    sys.exit(main())
