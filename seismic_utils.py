# seismic_utils.py

# Imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from devito import Operator, Eq, solve, Grid, SparseFunction
from devito import TimeFunction, Function, norm, gaussian_smooth, mmax
from devito.logger import info
from devito import configuration
from examples.seismic import (
    Model, plot_velocity, plot_shotrecord, Receiver, PointSource,
    plot_image, AcquisitionGeometry, TimeAxis,
    demo_model, plot_perturbation
)
from examples.seismic.self_adjoint import setup_w_over_q
from examples.seismic.acoustic import AcousticWaveSolver
import scipy

configuration['log-level'] = 'WARNING'


# User-defined functions
def generate_source_locations(case, nshots):
    source_locations1 = np.empty((nshots, 2), dtype=np.float32)
    source_locations1[:, 0] = np.linspace(0., 1000, num=nshots)
    source_locations1[:, 1] = 3.
    source_locations2 = np.empty((nshots, 2), dtype=np.float32)
    source_locations2[:, 0] = 3
    source_locations2[:, 1] = np.linspace(0., 1000, num=nshots)
    source_locations3 = np.empty((nshots, 2), dtype=np.float32)
    source_locations3[:, 0] = np.linspace(0., 1000, num=nshots)
    source_locations3[:, 1] = 497.
    source_locations4 = np.empty((nshots, 2), dtype=np.float32)
    source_locations4[:, 0] = 497
    source_locations4[:, 1] = np.linspace(0., 1000, num=nshots)

    options = {
        1: source_locations1, 2: source_locations2, 3: source_locations3, 4: source_locations4,
        12: np.concatenate((source_locations1, source_locations2)),
        13: np.concatenate((source_locations1, source_locations3)),
        14: np.concatenate((source_locations1, source_locations4)),
        23: np.concatenate((source_locations2, source_locations3)),
        24: np.concatenate((source_locations2, source_locations4)),
        34: np.concatenate((source_locations3, source_locations4)),
        123: np.concatenate((source_locations1, source_locations2, source_locations3)),
        124: np.concatenate((source_locations1, source_locations2, source_locations4)),
        134: np.concatenate((source_locations1, source_locations3, source_locations4)),
        234: np.concatenate((source_locations2, source_locations3, source_locations4)),
        1234: np.concatenate((source_locations1, source_locations2, source_locations3, source_locations4)),
    }

    if case not in options:
        raise ValueError("Invalid source case. Please choose a valid value.")
    return options[case]


def generate_receiver_coordinates(case, nreceivers, domain_size):
    rec_coordinates1 = np.empty((nreceivers, 2))
    rec_coordinates1[:, 1] = 0
    rec_coordinates1[:, 0] = np.linspace(0, domain_size[0], num=nreceivers)
    rec_coordinates2 = np.empty((nreceivers, 2))
    rec_coordinates2[:, 1] = np.linspace(0, domain_size[0], num=nreceivers)
    rec_coordinates2[:, 0] = 0
    rec_coordinates3 = np.empty((nreceivers, 2))
    rec_coordinates3[:, 1] = domain_size[0]
    rec_coordinates3[:, 0] = np.linspace(0, domain_size[0], num=nreceivers)
    rec_coordinates4 = np.empty((nreceivers, 2))
    rec_coordinates4[:, 1] = np.linspace(0, domain_size[1], num=nreceivers)
    rec_coordinates4[:, 0] = domain_size[0]

    options = {
        1: rec_coordinates1, 2: rec_coordinates2, 3: rec_coordinates3, 4: rec_coordinates4,
        12: np.concatenate((rec_coordinates1, rec_coordinates2)),
        13: np.concatenate((rec_coordinates1, rec_coordinates3)),
        14: np.concatenate((rec_coordinates1, rec_coordinates4)),
        23: np.concatenate((rec_coordinates2, rec_coordinates3)),
        24: np.concatenate((rec_coordinates2, rec_coordinates4)),
        34: np.concatenate((rec_coordinates3, rec_coordinates4)),
        123: np.concatenate((rec_coordinates1, rec_coordinates2, rec_coordinates3)),
        124: np.concatenate((rec_coordinates1, rec_coordinates2, rec_coordinates4)),
        134: np.concatenate((rec_coordinates1, rec_coordinates3, rec_coordinates4)),
        234: np.concatenate((rec_coordinates2, rec_coordinates3, rec_coordinates4)),
        1234: np.concatenate((rec_coordinates1, rec_coordinates2, rec_coordinates3, rec_coordinates4)),
    }

    if case not in options:
        raise ValueError("Invalid receiver case. Please choose a valid value.")
    return options[case]


def plot_shotrecord2(rec, model, t0, tn, colorbar=True, name=None):
    scale = np.max(rec) / 10.
    extent = [model.origin[0], model.origin[0] + 1e-3 * model.domain_size[0], 1e-3 * tn, t0]
    plot = plt.imshow(rec, vmin=-scale, vmax=scale, cmap='gray', extent=extent)
    plt.xlabel('Distance (km)')
    plt.ylabel('Time (s)')
    plt.title(name)
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)
    plt.show()


def compute_residual(residual, dobs, dsyn):
    if residual.grid.distributor.is_parallel:
        assert np.allclose(dobs.coordinates.data[:], dsyn.coordinates.data)
        assert np.allclose(residual.coordinates.data[:], dsyn.coordinates.data)
        diff_eq = Eq(residual, dsyn.subs({dsyn.dimensions[-1]: residual.dimensions[-1]}) -
                               dobs.subs({dobs.dimensions[-1]: residual.dimensions[-1]}))
        Operator(diff_eq)()
    else:
        residual.data[:] = dsyn.data[:] - dobs.data[:]
    return residual


def fwi_gradient(vp_in, true_model, geometry, nshots, source_locations, solver, mult):
    grad = Function(name="grad", grid=true_model.grid)
    residual = Receiver(name='residual', grid=true_model.grid,
                        time_range=geometry.time_axis, coordinates=geometry.rec_positions)
    d_obs = Receiver(name='d_obs', grid=true_model.grid,
                     time_range=geometry.time_axis, coordinates=geometry.rec_positions)
    d_syn = Receiver(name='d_syn', grid=true_model.grid,
                     time_range=geometry.time_axis, coordinates=geometry.rec_positions)
    objective = 0.
    for i in range(nshots * mult):
        geometry.src_positions[0, :] = source_locations[i, :]
        _, _, _ = solver.forward(vp=true_model.vp, rec=d_obs)
        _, u0, _ = solver.forward(vp=vp_in, save=True, rec=d_syn)
        compute_residual(residual, d_obs, d_syn)
        objective += .5 * norm(residual) ** 2
        solver.gradient(rec=residual, u=u0, vp=vp_in, grad=grad)
    return objective, grad


def plot_shotrecord_array(rec, model, t0, tn, case_rcv, colorbar=True, name=None):
    """
    Plot a shot record (receiver values over time).

    Parameters
    ----------
    rec :
        Receiver data with shape (time, points).
    model : Model
        object that holds the velocity model.
    t0 : int
        Start of time dimension to plot.
    tn : int
        End of time dimension to plot.
    case_rcv : int
        Indicates the location of the source and receiver array:
        1 - Top
        2 - Left
        3 - Bottom
        4 - Right
    colorbar : bool, optional
        Whether to show the colorbar. Default is True.
    name : str, optional
        Title for the entire figure. Default is 'True data'.
    """
    scale = np.max(rec) / 10.
    extent = [model.origin[0], model.origin[0] + 1e-3*model.domain_size[0],
              1e-3*tn, t0]

    num_plots = len(str(case_rcv))
    rows = 1
    cols = num_plots

    fig, axes = plt.subplots(rows, cols, figsize=(5*num_plots, 5))
    if num_plots == 1:
        axes = [axes] 

    fig.suptitle(name, fontsize=16, fontweight='bold', y=0.98, ha='center')  # Add general title

    for i in range(num_plots):
        rec_slice = slice_shotrecord(rec, case_rcv % 10)
        if case_rcv % 10 == 1:
            axes[i].imshow(rec_slice, vmin=-scale, vmax=scale, cmap='gray', extent=extent)
            axes[i].set_title('Top receivers')
        elif case_rcv % 10 == 2:
            axes[i].imshow(rec_slice, vmin=-scale, vmax=scale, cmap='gray', extent=extent)
            axes[i].set_title('Left receivers')
        elif case_rcv % 10 == 3:
            axes[i].imshow(rec_slice, vmin=-scale, vmax=scale, cmap='gray', extent=extent)
            axes[i].set_title('Bottom receivers')
        elif case_rcv % 10 == 4:
            axes[i].imshow(rec_slice, vmin=-scale, vmax=scale, cmap='gray', extent=extent)
            axes[i].set_title('Right receivers')
        axes[i].set_xlabel('X position (km)')
        if i == 0:  # Only set y-label for the first subplot
            axes[i].set_ylabel('Time (s)')
        case_rcv = case_rcv // 10

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca() if num_plots == 1 else axes[-1]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(ax.images[0], cax=cax)
    plt.show()


def plot_velocity_clim(model, source=None, receiver=None, colorbar=True, cmap=None, vmin=None, vmax=None, name=None):
    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]
    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))
    field = model.vp.data[slices] if hasattr(model, 'vp') else model.lam.data[slices]
    vmin = np.min(field) if vmin is None else vmin
    vmax = np.max(field) if vmax is None else vmax
    fig = plt.figure(figsize=(6, 3), dpi=150)
    plot = plt.imshow(np.transpose(field), cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (km)')
    plt.title(name)
    if receiver is not None:
        plt.scatter(1e-3 * receiver[:, 0], 1e-3 * receiver[:, 1], s=25, c='green', marker='D')
    if source is not None:
        plt.scatter(1e-3 * source[:, 0], 1e-3 * source[:, 1], s=25, c='red', marker='o')
    plt.xlim(model.origin[0], model.origin[0] + domain_size[0])
    plt.ylim(model.origin[1] + domain_size[1], model.origin[1])
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label('Velocity (km/s)')
    plt.show()

def compute_update(geometry, solver, start_model, true_model, src_coordinates, nshots, shape, nbl, mult):
    ff, update = fwi_gradient(
        vp_in=start_model.vp,
        true_model=true_model,
        geometry=geometry,
        nshots=nshots,
        source_locations=src_coordinates,
        solver=solver,
        mult=mult
    )
    alpha = 0.5 / mmax(update)
    a = -update.data[nbl:shape[0]+nbl, nbl:shape[1]+nbl]
    b = (true_model.vp.data - start_model.vp.data)[nbl:shape[0]+nbl, nbl:shape[1]+nbl]
    c = (start_model.vp.data + alpha * update.data)[nbl:shape[0]+nbl, nbl:shape[1]+nbl]
    return ff, a, b, c

def slice_shotrecord(rec, case):
    """
    Slice the shot record data based on the case.

    Parameters
    ----------
    rec : numpy.ndarray
        Receiver data with shape (time, points).
    case : int
        Indicates the location of the source and receiver array:
        1 - Top
        2 - Left
        3 - Bottom
        4 - Right

    Returns
    -------
    numpy.ndarray
        Sliced shot record data.
    """
    num_columns = rec.shape[1]
    if case == 1:
        return rec[:, :num_columns // 4]
    elif case == 2:
        return rec[:, num_columns // 4 : num_columns // 2]
    elif case == 3:
        return rec[:, num_columns // 2 : 3 * num_columns // 4]
    elif case == 4:
        return rec[:, 3 * num_columns // 4:]