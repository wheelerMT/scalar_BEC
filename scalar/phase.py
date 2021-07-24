from grid import Grid
import cupy as cp
import numpy as np


def _generate_random_pos(nvort: int, thresh: float, grid: Grid):
    accepted_pos = []
    iterations = 0
    while len(accepted_pos) < nvort:
        within_range = True
        while within_range:
            iterations += 1
            triggered = False

            # Generate a position
            pos = cp.random.uniform(-grid.len_x // 2, grid.len_x // 2), \
                  cp.random.uniform(-grid.len_y // 2, grid.len_y // 2)

            # Check if position is too close to any other position
            for accepted_position in accepted_pos:
                if abs(pos[0] - accepted_position[0]) < thresh:
                    if abs(pos[1] - accepted_position[1]) < thresh:
                        triggered = True
                        break

            # If position isn't close to others then add it to accepted positions
            if not triggered:
                accepted_pos.append(pos)
                within_range = False

        # Prints out current progress every 500 iterations
        if len(accepted_pos) % 500 == 0:
            print('Found {} positions...'.format(len(accepted_pos)))

    print('Found {} positions in {} iterations.'.format(len(accepted_pos), iterations))
    return iter(accepted_pos)  # Set accepted positions to member


class Phase:
    def __init__(self, nvort: int, thresh: float, grid: Grid, phase_type: str):
        self.phase = None

        if phase_type == 'random':
            # If random is chosen, generate random positions then imprint
            initial_pos = _generate_random_pos(nvort, thresh, grid)
            self._imprint_phase(nvort, grid, initial_pos)

    def _imprint_phase(self, nvort: int, grid: Grid, pos: iter):
        # Initialise phase:
        theta_tot = cp.empty((grid.Nx, grid.Ny))

        # Scale pts:
        x_tilde = 2 * cp.pi * ((grid.X - grid.X.min()) / grid.len_x)
        y_tilde = 2 * cp.pi * ((grid.Y - grid.Y.min()) / grid.len_y)

        # Construct phase for this postion:
        for _ in range(nvort):
            theta_k = cp.zeros((grid.Nx, grid.Ny))

            try:
                x_m, y_m = next(pos)
                x_p, y_p = next(pos)
            except StopIteration:
                break

            # Scaling vortex positions:
            x_m_tilde = 2 * cp.pi * ((x_m - grid.X.min()) / grid.len_x)
            y_m_tilde = 2 * cp.pi * ((y_m - grid.Y.min()) / grid.len_y)
            x_p_tilde = 2 * cp.pi * ((x_p - grid.X.min()) / grid.len_x)
            y_p_tilde = 2 * cp.pi * ((y_p - grid.Y.min()) / grid.len_y)

            # Aux variables
            Y_minus = y_tilde - y_m_tilde
            X_minus = x_tilde - x_m_tilde
            Y_plus = y_tilde - y_p_tilde
            X_plus = x_tilde - x_p_tilde

            heav_xp = cp.asarray(np.heaviside(cp.asnumpy(X_plus), 1.))
            heav_xm = cp.asarray(np.heaviside(cp.asnumpy(X_minus), 1.))

            for nn in cp.arange(-5, 6):
                theta_k += cp.arctan(cp.tanh((Y_minus + 2 * cp.pi * nn) / 2) * cp.tan((X_minus - cp.pi) / 2)) \
                           - cp.arctan(cp.tanh((Y_plus + 2 * cp.pi * nn) / 2) * cp.tan((X_plus - cp.pi) / 2)) \
                           + cp.pi * (heav_xp - heav_xm)

            theta_k -= y_tilde * (x_p_tilde - x_m_tilde) / (2 * cp.pi)
            theta_tot += theta_k

        self.phase = theta_tot
