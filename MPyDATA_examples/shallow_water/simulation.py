from MPyDATA.factories import Factories
from MPyDATA_examples.shallow_water.setup import Setup
from MPyDATA import Options
from auxiliary import extrapolate_in_time, interpolate_in_space, grad
from MPyDATA import ScalarField, VectorField, PeriodicBoundaryCondition
import numpy as np
from mpdata import mpdata_wrapper



class Simulation:
    def __init__(self, setup: Setup, options: Options):
        setup = Setup()
        x = setup.grid
        state = setup.H0(x)

        self.stepper = Factories.shallow_water(
            nr = len(setup.grid),
            r_min = min(setup.grid),
            r_max = max(setup.grid),
            data=state,
            C = setup.C,
            opts=options)
        self.nt = setup.nt

    @property
    def state(self):
        return self.stepper.curr.get().copy()

    def run(self):
        self.stepper.advance(self.nt)

    def run_1d(self, m_value =.2, dx = 0.05, dt = .01, nx = 16, n_steps = 20):
        """
        @author: Paweł Rozwoda
        """
        # initial state of liquid

        initial_m_value = m_value
        xc = int(nx / 2)
        shift = 4
        grid = np.linspace(0, nx, nx)
        H0 = lambda x: np.where(abs(x - xc) < 1, 1 - (x - xc) ** 2, 0)
        h0 = H0(grid)
        h_initial = np.full(nx, initial_m_value)
        # h_initial[xc-shift:xc+shift] = np.array([0.8 - (0.02 * i) for i in range(2*shift)])
        h_initial = h0

        h_old = np.array(h_initial)
        h_new = np.array(h_initial)

        uh_new = np.zeros(nx)
        uh_old = np.zeros(nx)

        # initial movement vector
        u_initial = np.zeros((nx + 1))
        u_old = np.array(u_initial)
        u_new = np.array(u_initial)

        rhs_old = np.array(h_old)
        rsh_old = np.zeros(nx)

        options = Options(n_iters=2, infinite_gauge=True, flux_corrected_transport=True)
        halo = options.n_halo

        advectee = ScalarField(
            data=h_new,
            halo=halo,
            boundary_conditions=(PeriodicBoundaryCondition(),)
        )

        advector = VectorField(
            data=(u_initial,),
            halo=halo,
            boundary_conditions=(PeriodicBoundaryCondition(), )
        )

        MPDATA = mpdata_wrapper(advector, advectee, (nx,))

        g = 1  # m/s^2
        for i in range(n_steps):
            u_mid = extrapolate_in_time(interpolate_in_space(uh_new, h_new), interpolate_in_space(uh_old, h_old))

            # RHS = 0, mass conservation
            h_new[:] = MPDATA(u_mid, h_old)
            rhs_new = -g * h_new * grad(h_new, dx=dx)

            # momentum conservation
            uh_new[:] = MPDATA(u_mid, uh_old + .5 * dt * rhs_old) + .5 * dt * rhs_new

            # replace variables
            h_old, h_new = h_new, h_old
            u_old, u_new = u_new, u_old
            rhs_old, rhs_new = rhs_new, rhs_old
            uh_old, uh_new = uh_new, uh_old

        return h_old



    def run_2d(self, n_steps = 20):
        """
        @author: Paweł Rozwoda
        """
        options = Options(n_iters=2, infinite_gauge=True, flux_corrected_transport=True)

        nx, ny = 10, 10
        dx = 1.
        dy = 1.
        dt = 0.05

        # initial state of liquid
        initial_m_value = 0.2
        h_initial = np.full((ny, nx), initial_m_value)

        # simple modification of liquid
        h_initial[4:8, 5:8] = np.array([[0.8 - (0.09 * i * j) for i in range(3)] for j in range(4)])

        h_old = np.array(h_initial)
        h_new = np.array(h_initial)

        uh_new_x = np.zeros((ny, nx))
        uh_old_x = np.zeros((ny, nx))
        uh_new_y = np.zeros((ny, nx))
        uh_old_y = np.zeros((ny, nx))

        u_initial_x = np.zeros((ny, nx + 1))
        u_initial_y = np.zeros((ny + 1, nx))

        u_old = np.array(u_initial_y), np.array(u_initial_x)
        u_new = np.array(u_initial_y), np.array(u_initial_x)

        rhs_old_x = np.array(h_initial)
        rhs_old_y = np.array(h_initial)
        rhs_new_x = np.array(h_initial)
        rhs_new_y = np.array(h_initial)

        halo = options.n_halo

        advectee = ScalarField(
            data=h_new,
            halo=halo,
            boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition())
        )

        advector = VectorField(
            data=(u_initial_y, u_initial_x),
            halo=halo,
            boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition())
        )

        MPDATA = mpdata_wrapper(advector, advectee, (ny, nx))

        g = 9.8  # m/s^2

        for i in range(n_steps):
            u_mid_x = extrapolate_in_time(interpolate_in_space(uh_new_x, h_new, axis='x'),
                                          interpolate_in_space(uh_old_x, h_old, axis='x'))
            u_mid_y = extrapolate_in_time(interpolate_in_space(uh_new_y, h_new, axis='y'),
                                          interpolate_in_space(uh_old_y, h_old, axis='y'))
            u_mid = (u_mid_y, u_mid_x)

            # RHS = 0, mass conservation
            h_new[:] = MPDATA(u_mid, h_old)
            grad_y, grad_x = grad(h_new, dx=dx, dy=dy)

            # without Coriolis parameter
            rhs_new_x = -g * h_new * grad_x + ((1. / h_new) * (grad_y - grad_x) * uh_new_y)
            rhs_new_y = -g * h_new * grad_y - ((1. / h_new) * (grad_y - grad_x) * uh_new_x)

            # momentum conservation
            uh_new_x[:] = MPDATA(u_mid, uh_old_x + .5 * dt * rhs_old_x) + .5 * dt * rhs_new_x
            uh_new_y[:] = MPDATA(u_mid, uh_old_y + .5 * dt * rhs_old_y) + .5 * dt * rhs_new_y

            # replace variables
            h_old, h_new = h_new, h_old
            u_old, u_new = u_new, u_old

            rhs_old_x, rhs_new_x = rhs_new_x, rhs_old_x
            rhs_old_y, rhs_new_y = rhs_new_y, rhs_old_y

            uh_old_x, uh_new_x = uh_new_x, uh_old_x
            uh_old_y, uh_new_y = uh_new_y, uh_old_y

        return h_old


    def err_1d(self, h_numerical, time, nx):
        grid = np.linspace(0, nx, nx)
        setup = Setup()
        H = lambda t: setup.analytic_H(grid, t, xc=nx/2)
        u = lambda t: setup.analytic_u(grid, t, xc=nx/2)
        h_analytical = H(time)
        h_diff = h_numerical - h_analytical
        points_nr = h_numerical.shape[0]
        if time==0:
            return 0
        else:
            return 1 / time * np.sqrt(((h_diff ** 2).sum() / points_nr))

