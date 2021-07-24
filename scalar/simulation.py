from grid import Grid
from wavefunction import Wavefunction
import cupy as cp


def _renormalise_atom_num(wfn: Wavefunction):
    wfn.psi_k = cp.fft.fft2(cp.sqrt(wfn.atom_number) * cp.fft.ifft2(wfn.psi_k)
                             / cp.sqrt(wfn.calc_atom_num(k_space=True)))


def _fix_phase(wfn: Wavefunction):
    wfn.psi = cp.fft.ifft2(wfn.psi_k)
    wfn.psi *= cp.exp(1j * wfn.phase) / cp.exp(1j * cp.angle(wfn.psi))
    wfn.psi_k = cp.fft.fft2(wfn.psi)


class Simulation:
    def __init__(self, dt: float, nframe: int):
        self.dt = dt
        self.nframe = nframe

    def _kinetic_step(self, wfn: Wavefunction, Kgrid: Grid):
        wfn.psi_k *= cp.exp(-0.25 * 1j * self.dt * Kgrid.squared)

    def _potential_step(self, wfn: Wavefunction):
        wfn.psi *= cp.exp(-1j * self.dt * (wfn.V + wfn.g * cp.abs(wfn.psi) ** 2))

    def imaginary_time(self, wfn: Wavefunction, Kgrid: Grid, nt: int):
        """Do split-step imaginary time evolution"""

        self.dt *= -1j  # Switch to imaginary time

        # Do imaginary time evolution
        for i in range(nt):
            self._kinetic_step(wfn, Kgrid)  # First Kinetic step

            wfn.psi = cp.fft.ifft2(wfn.psi_k)  # Inverse FFT

            self._potential_step(wfn)   # Potential step

            wfn.psi_k = cp.fft.fft2(wfn.psi)    # Forward FFT

            self._kinetic_step(wfn, Kgrid)  # Last kinetic step

            _renormalise_atom_num(wfn)

            _fix_phase(wfn)

            # Print current time:
            if i % 50 == 0:
                print(f"t = {i * self.dt}")

        self.dt *= 1j   # Switch back to real time

    def real_time(self, wfn: Wavefunction, Kgrid: Grid, nt: int):
        """Do split-step real time evolution"""

        # Do real time evolution
        for i in range(nt):
            self._kinetic_step(wfn, Kgrid)  # First Kinetic step

            wfn.psi = cp.fft.ifft2(wfn.psi_k)  # Inverse FFT

            self._potential_step(wfn)  # Potential step

            wfn.psi_k = cp.fft.fft2(wfn.psi)  # Forward FFT

            self._kinetic_step(wfn, Kgrid)  # Last kinetic step

            # Print current time:
            if i % 50 == 0:
                print(f"t = {i * self.dt}")
