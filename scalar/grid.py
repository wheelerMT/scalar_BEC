import cupy as cp


class Grid:
    """Generates a grid object for use with the wavefunction. Contains the properties
    of the grid as well as the 2D meshgrids themselves.

    Attributes
    ----------
    nx : int
        Number of x grid points.
    ny : int
        Number of y grid points.
    dx : float
        x grid spacing.
    dy : float
        y grid spacing.
    len_x : float
        Length of grid in x-direction.
    len_y : float
        Length of grid in y-direction.
    X : ndarray
        2D :obj:`ndarray` of X meshgrid.
    Y : ndarray
        2D :obj:`ndarray` of Y meshgrid.
    squared : ndarray
        2D :obj:`ndarray` storing result of X ** 2 + Y ** 2.
    """

    def __init__(self, nx: int, ny: int, dx: float = 1., dy: float = 1.):
        """Instantiate a Grid object.
        Automatically generates meshgrids using parameters provided.

        Parameters
        ----------
        nx : int
            Number of x grid points.
        ny : int
            Number of y grid points.
        dx : float
            Grid spacing for x grid, defaults to 1.
        dy : float
            Grid spacing for y grid, defaults to 1.
        """

        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy
        self.len_x, self.len_y = nx * dx, ny * dy

        # Generate 2D meshgrids:
        self.X, self.Y = cp.meshgrid(cp.arange(-nx // 2, nx // 2) * dx, cp.arange(-ny // 2, ny // 2) * dy)

        self.squared = self.X ** 2 + self.Y ** 2

    def fftshift(self):
        """Performs FFT shift on X & Y meshgrids.
        """

        self.X = cp.fft.fftshift(self.X)
        self.Y = cp.fft.fftshift(self.Y)
