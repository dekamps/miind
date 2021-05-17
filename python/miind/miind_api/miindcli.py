import click
import matplotlib
import matplotlib.pyplot as plt
import copy

from MiindSimulation import MiindSimulation

@click.group(invoke_without_command=True, context_settings=dict(ignore_unknown_options=True,allow_extra_args=True,))
@click.argument("xml_path", type=click.Path(exists=True))
@click.option("--directory", "-d", type=click.Path(exists=False))
@click.pass_context
def cli(ctx, xml_path, **kwargs):
    print kwargs
    ctx.obj = MiindSimulation(xml_path, kwargs['directory'])
    pass

@cli.command("submit", short_help='Generate a MIIND executable based on a ' +
             '"xml" parameter file.')
@click.option("--directory", "-d", type=click.Path(exists=True))
@click.pass_context
def generate_(ctx, xml_path, **kwargs):
    sim = ctx.obj
    sim.submit()

def main():
    cli()

if __name__ == "__main__":
    main()
