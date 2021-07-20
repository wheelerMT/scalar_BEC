import cupy as cp


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
        self.X, self.Y = cp.meshgrid(cp.arange(-Nx // 2, Nx // 2) * dx, cp.arange(-Ny // 2, Ny // 2) * dy)

        self.squared = self.X ** 2 + self.Y ** 2

    def fftshift(self):
        """
        Performs FFT shift on meshgrids.
        """
        self.X = cp.fft.fftshift(self.X)
        self.Y = cp.fft.fftshift(self.Y)
