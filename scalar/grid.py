import numpy as np


class Grid:
    def __init__(self, Nx: int, Ny: int, dx: float = 1., dy: float = 1.) -> None:
        """

        :param Nx: Number of x grid points.
        :param Ny: Number of y grid points.
        :param dx: x grid spacing.
        :param dy: y grid spacing.
        """

        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = dx, dy
        self.len_x, self.len_y = Nx * dx, Ny * dy

        # Generate 2D meshgrids:
        self.X, self.Y = np.meshgrid(np.arange(-Nx // 2, Nx // 2) * dx, np.arange(-Ny // 2, Ny // 2) * dy)

    def fftshift(self):
        """
        Performs FFT shift on meshgrids.
        """
        self.X = np.fft.fftshift(self.X)
        self.Y = np.fft.fftshift(self.Y)
