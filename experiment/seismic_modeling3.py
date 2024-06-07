# referred to https://www.devitoproject.org/examples/seismic/tutorials/01_modelling.html

import matplotlib.pyplot as plt
import numpy as np
from devito import Eq, Function, Max, Min, Operator, mmax, norm, set_log_level, TimeFunction, Inc, solve
from lib.seismic import AcquisitionGeometry, Receiver, demo_model, plot_velocity
from lib.seismic.acoustic import AcousticWaveSolver

# devitoのlogの抑制
set_log_level("WARNING")

nshots = 9  # Number of shots to create gradient from
nreceivers = 101  # Number of receiver locations per shot
fwi_iterations = 5  # Number of outer FWI iterations

shape = (101, 101)  # Number of grid point (nx, nz)
spacing = (10.0, 10.0)  # Grid spacing in m. The domain size is now 1km by 1km
origin = (0.0, 0.0)  # Need origin to define relative source and receiver locations

model = demo_model("circle-isotropic", vp_circle=3.0, vp_background=2.5, origin=origin, shape=shape, spacing=spacing, nbl=40)

model0 = demo_model("circle-isotropic", vp_circle=2.5, vp_background=2.5, origin=origin, shape=shape, spacing=spacing, nbl=40, grid=model.grid)

assert model.grid == model0.grid
assert model.vp.grid == model0.vp.grid

t0 = 0.0
tn = 1000.0
f0 = 0.010

src_coordinates = np.empty((1, 2))
src_coordinates[0, :] = np.array(model.domain_size) * 0.5
src_coordinates[0, 0] = 20.0  # Depth is 20m

rec_coordinates = np.empty((nreceivers, 2))
rec_coordinates[:, 1] = np.linspace(0, model.domain_size[0], num=nreceivers)
rec_coordinates[:, 0] = 980.0

geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type="Ricker")

solver = AcousticWaveSolver(model, geometry, space_order=4)
true_d, _, _ = solver.forward(vp=model.vp)
smooth_d, _, _ = solver.forward(vp=model0.vp)

# Prepare the varying source locations sources
source_locations = np.empty((nshots, 2), dtype=np.float32)
source_locations[:, 0] = 30.0
source_locations[:, 1] = np.linspace(0.0, 1000, num=nshots)


# Computes the residual between observed and synthetic data into the residual
def compute_residual(residual, dobs, dsyn):
    residual.data[:] = dsyn.data[:] - dobs.data[:]
    return residual


def get_grad_op():
    grad = Function(name='grad', grid=model.grid)
    u = TimeFunction(name='u', grid=model.grid, save=geometry.nt, time_order=2, space_order=solver.space_order)
    v = TimeFunction(name='v', grid=model.grid, save=None, time_order=2, space_order=solver.space_order)

    eqns = [Eq(v.backward, solve(model.m * v.dt2 - v.laplace + model.damp * v.dt.T, v.backward), subdomain=model.grid.subdomains['physdomain'])]
    rec_term = geometry.rec.inject(field=v.backward, expr=geometry.rec * model.grid.stepping_dim.spacing**2 / model.m)
    gradient_update = Inc(grad, - u * v.dt2)

    return Operator(eqns + rec_term + [gradient_update], subs=model.spacing_map, name='Gradient', **solver._kwargs)


grad_op = get_grad_op()


# Create FWI gradient kernel
def fwi_gradient(vp_in):

    grad = Function(name="grad", grid=model.grid)
    residual = Receiver(name="residual", grid=model.grid, time_range=geometry.time_axis, coordinates=geometry.rec_positions)
    observed_waveform = Receiver(name="d_obs", grid=model.grid, time_range=geometry.time_axis, coordinates=geometry.rec_positions)
    calculated_waveform = Receiver(name="d_syn", grid=model.grid, time_range=geometry.time_axis, coordinates=geometry.rec_positions)
    objective = 0.0

    for i in range(nshots):
        geometry.src_positions[0, :] = source_locations[i, :]
        solver.forward(vp=model.vp, rec=observed_waveform)
        _, calculated_wave_field, _ = solver.forward(vp=vp_in, save=True, rec=calculated_waveform)
        residual.data[:] = calculated_waveform.data[:] - observed_waveform.data[:]

        objective += 0.5 * norm(residual) ** 2

        v = TimeFunction(name="v", grid=solver.model.grid, time_order=2, space_order=solver.space_order)

        grad_op.apply(rec=residual, grad=grad, v=v, u=calculated_wave_field, dt=solver.dt, vp=vp_in)

    return objective, grad


# Compute gradient of initial model
ff, update = fwi_gradient(model0.vp)
assert np.isclose(ff, 57283, rtol=1e0)

# Show what the update does to the model
alpha = 0.5 / mmax(update)


# Define bounding box constraints on the solution.
def update_with_box(vp, alpha, dm, vmin=2.0, vmax=3.5):
    """
    Apply gradient update in-place to vp with box constraint

    Notes:
    ------
    For more advanced algorithm, one will need to gather the non-distributed
    velocity array to apply constrains and such.
    """
    update = vp + alpha * dm
    update_eq = Eq(vp, Max(Min(update, vmax), vmin))
    Operator(update_eq)()


# Run FWI with gradient descent
history = np.zeros((fwi_iterations, 1))
for i in range(0, fwi_iterations):
    # Compute the functional value and gradient for the current
    # model estimate
    phi, direction = fwi_gradient(model0.vp)

    # Store the history of the functional values
    history[i] = phi

    # Artificial Step length for gradient descent
    # In practice this would be replaced by a Linesearch (Wolfe, ...)
    # that would guarantee functional decrease Phi(m-alpha g) <= epsilon Phi(m)
    # where epsilon is a minimum decrease constant
    alpha = 0.05 / mmax(direction)

    # Update the model estimate and enforce minimum/maximum values
    update_with_box(model0.vp, alpha, direction)

    # Log the progress made
    print("Objective value is %f at iteration %d" % (phi, i + 1))

# Plot inverted velocity model
plot_velocity(model0)


# Plot objective function decrease
# plt.figure()
# plt.loglog(history)
# plt.xlabel("Iteration number")
# plt.ylabel("Misift value Phi")
# plt.title("Convergence")
# plt.show()
