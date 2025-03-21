# MATH
try:
    from jax import numpy as np
    from jax import Array
    from jax.lax import fori_loop
except ImportError:
    import numpy as np
    from numpy import ndarray as Array

    fori_loop = None

from numpy import loadtxt, ndarray

# PLOTTING
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# ASE / PYMATGEN
from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.calculators.calculator import Calculator
from pymatgen.io.vasp import Incar

# MISCELLANEUS
from tqdm import tqdm
from typing import Optional


class VaspMDAnalyzer:
    __atoms_posis: Array
    __cells: Array
    __atoms: dict[str, int]
    __potim: float  # fs

    __msd: dict = dict()
    __D: dict = dict()

    def __init__(
        self,
        xdatcar_path: str = "./XDATCAR",
        potim: float = 1,
        start_conf: int = 0,
        nconf: Optional[int] = None,
        jump_elimination: bool = True,
    ) -> None:
        # Save potim
        self.__potim = potim

        # Retrive informations on the atomic species
        self.__get_atoms_xdatcar(xdatcar_path)
        n_atoms = sum(self.__atoms.values())

        # Look if simulation left cells parameters unchanged
        read_cells = self.__is_cell_printed(xdatcar_path)

        # Collect atom positions and unit cell at every frame
        cells, posis = [], []

        print("VaspMDAnalyzer: reading XDATCAR file...")
        with open(xdatcar_path, "r") as data:
            # if cell is fixed then cell is reported in first frame
            if not read_cells:
                cells.append(loadtxt(data, skiprows=2, max_rows=3))
                for _ in range(start_conf * (1 + n_atoms) + 2):
                    data.readline()
            else:
                for _ in range(start_conf * (8 + n_atoms)):
                    data.readline()

            # Read all other frames
            if nconf is None:
                while not data.readline() == "":
                    if read_cells:
                        cells.append(loadtxt(data, skiprows=1, max_rows=3))
                        posis.append(loadtxt(data, skiprows=3, max_rows=n_atoms))
                    else:
                        posis.append(loadtxt(data, max_rows=n_atoms))
            else:
                for _ in tqdm(range(nconf)):
                    if data.readline() == "":
                        break

                    if read_cells:
                        cells.append(loadtxt(data, skiprows=1, max_rows=3))
                        posis.append(loadtxt(data, skiprows=3, max_rows=n_atoms))
                    else:
                        posis.append(loadtxt(data, max_rows=n_atoms))

        self.__cells, self.__atoms_posis = np.array(cells), np.array(posis)

        if not read_cells:
            self.__cells = np.repeat(self.__cells, self.__atoms_posis.shape[0], axis=0)

        # --Data postprocessing
        self.__cart_transform(jump_elimination)
        self.__remove_drift()  # remove the drift of the cell

        print("VaspMDAnalyzer: XDATCAR read succesfully!")

    def get_total_frame(self) -> int:
        return self.__atoms_posis.shape[0]

    def get_atomic_position(self, element: Optional[str] = None) -> Array:
        if element is None:
            return self.__atoms_posis

        count = 0
        for key, value in self.__atoms.items():
            if key == element.strip():
                return self.__atoms_posis[:, count : count + value, :]
            count += value

        temp = " ".join(self.__atoms.keys())
        raise IndexError(
            f"Element {element} not present in the system, possible choices were: {temp}"
        )

    def plot_MSD(
        self, element: str, ax: Optional[Axes] = None, scale: Optional[str] = None
    ) -> None:
        show = ax is None

        if show:
            ax = plt.subplot()

        y = self.get_MSD(element, method="lax")
        x = self.__potim * np.arange(y.shape[0])

        ax.plot(x, y[:, 0], "--", label="X-component", color="darkcyan")
        ax.plot(x, y[:, 1], "-.", label="Y-component", color="slateblue")
        ax.plot(x, y[:, 2], ":", label="Z-component", color="seagreen")
        ax.plot(x, y.sum(1), "-", label="Total", color="black")

        ax.set_ylabel(r"MSD ($\AA^2$)", fontsize=16)
        ax.set_xlabel(r"time (fs)", fontsize=16)

        ax.legend()

        if scale is not None:
            ax.set_yscale(scale)
            ax.set_xscale(scale)

        if show:
            plt.show()

    def get_atoms_from_frame(self, frame: int) -> Atoms:
        symbols = []
        for element, n in self.__atoms.items():
            for _ in range(n):
                symbols.append(element)

        return Atoms(
            symbols,
            positions=self.__atoms_posis[frame],
            cell=self.__cells[frame],
            pbc=True,
        )

    def get_diffusion_coefficient(self, element: str):
        msd = self.get_MSD(element, method="lax")

        D = (np.roll(msd, -1, 0) - np.roll(msd, 1, 0)) / (2 * self.__potim)

        if element is not None:
            self.__D[element] = D

        return D

    def get_MSD(
        self, element: str, method: str = "fft", recompute: bool = False
    ) -> Array:
        if element in self.__msd.keys() and not recompute:
            return self.__msd[element]

        # Possible errors in element selection are handled here
        msd = self.get_atomic_position(element)

        if method.strip().lower() == "base":
            msd = self.vectorized_msd(msd)
        elif method.strip().lower() == "lax":
            msd = self.lax_msd(msd)
        elif method.strip().lower() == "fft":
            msd = self.fft_msd(msd)
        elif method.strip().lower() == "fft_lax":
            msd = self.fft_lax_msd(msd)
        else:
            raise NotImplementedError(
                f"The method {method} selected is not implemented"
            )

        if element is not None:
            self.__msd[element] = msd

        return msd

    @staticmethod
    def fft_lax_msd(atomic_positions: Array) -> Array:
        if fori_loop is None or Array is ndarray:
            raise NotImplementedError("lax methods are avaliable only with jax!")

        N = atomic_positions.shape[0]  # Number of frames

        D = np.append(
            np.square(atomic_positions), np.zeros_like(atomic_positions[0:1]), 0
        )
        Q = 2 * D.sum(0)

        def comp_s(i, val):
            Q, S = val
            Q -= D[i - 1] + D[N - i]
            return Q, S.at[i].set(Q)

        Q, S1 = fori_loop(0, N, comp_s, (Q, np.zeros_like(atomic_positions)))

        S2 = np.fft.fft(atomic_positions, 2 * N, axis=0)
        S2 = np.fft.ifft(S2 * S2.conjugate(), axis=0)[:N].real

        return (S1 - 2 * S2).mean(1) / (N - np.arange(N).reshape(N, 1))

    @staticmethod
    def fft_msd(atomic_positions: Array) -> Array:
        N = atomic_positions.shape[0]  # Number of frames

        D = np.square(atomic_positions)
        D1 = np.append(np.zeros_like(D[0:1]), D, 0)
        D2 = np.append(D, np.zeros_like(D[0:1]), 0)

        S1 = (2 * D.sum(0, keepdims=True) - np.cumsum(D1 + np.flip(D2, 0), 0))[:-1]

        S2 = np.fft.fft(atomic_positions, 2 * N, axis=0)
        S2 = np.fft.ifft(S2 * S2.conjugate(), axis=0)[:N].real

        return ((S1 - 2 * S2) / (N - np.arange(N).reshape(N, 1, 1))).mean(1)

    @staticmethod
    def vectorized_msd(atomic_positions: Array) -> Array:
        N = atomic_positions.shape[0]  # Number of frames

        msd = []
        for shift in range(N):
            rshift = np.roll(atomic_positions, -shift, 0)
            idx, _, _ = np.indices(atomic_positions.shape)

            msd.append(
                np.where(idx < N - shift, np.square(rshift - atomic_positions), 0).sum(
                    0
                )
                / (N - shift)
            )

        return np.array(msd).mean(1)

    @staticmethod
    def lax_msd(atomic_positions: Array) -> Array:
        if fori_loop is None or Array is ndarray:
            raise NotImplementedError("lax methods are avaliable only with jax!")

        N = atomic_positions.shape[0]  # Number of frames

        def compute(shift: int, msd: Array):
            res = np.roll(atomic_positions, -shift, 0)

            frame, _, _ = np.indices(atomic_positions.shape)
            res = np.where(frame < N - shift, res - atomic_positions, 0)

            res = np.square(res).sum(0) / (N - shift)

            return msd.at[shift].set(res.mean(0))

        return fori_loop(
            0,
            N,
            compute,
            np.zeros((N, 3)),
        )

    def write(self, path: str):
        atoms = self.__get_atoms_string()

        with open(path, "w") as file:
            for i, (cell, pos) in tqdm(
                enumerate(zip(self.__cells, self.__atoms_posis))
            ):
                file.write("unknown system\n\t1\n")

                cel_line = "\n".join(
                    map(
                        lambda x: f"   {x[0]:10.6f}   {x[1]:10.6f}   {x[2]:10.6f}",
                        cell.tolist(),
                    )
                )

                file.write(cel_line)
                file.write("\n" + atoms)
                file.write(f"Cartesian configuration= {i+1}\n")

                pos_line = "\n".join(
                    map(
                        lambda x: f"   {x[0]:10.8f}  {x[1]:10.8f}  {x[2]:10.8f}",
                        pos.tolist(),
                    )
                )
                file.write(pos_line + "\n")

    def __cart_transform(self, jump_elimination: bool) -> None:
        # --If no jump elimination is required then just transform
        if not jump_elimination:
            self.__atoms_posis = np.einsum(
                "ijk,ikl->ijl", self.__atoms_posis, self.__cells
            )
            return

        # --Search for jumps in direct coordinates
        # Compute difference in position between frames
        variation = np.diff(self.__atoms_posis, axis=0)

        # Look where the atoms move more than 0.5 a unit vector
        jump_right = variation > 0.5
        jump_left = variation < -0.5

        # Dected the frame where the jumps occures
        which_frame = np.any(np.logical_or(jump_right, jump_left), axis=(1, 2))
        which_frame = np.arange(which_frame.shape[0])[which_frame]

        # --Transform in cartesian
        self.__atoms_posis = np.einsum("ijk,ikl->ijl", self.__atoms_posis, self.__cells)

        # --Eliminates the jumps
        # Separate the cell's unit vector for every frame
        unit_cell_vectors = np.split(self.__cells, 3, 1)

        # Go through the left or right jump and recall if you have to add or remove a unit vector
        for jump, mod in zip([jump_right, jump_left], [-1, +1]):
            # Split the direct components to see where the jump occures
            jump_and_vector = zip(np.split(jump, 3, 2), unit_cell_vectors)

            # Loop over a tuple having:
            # 1 - (jump_comp) list of bool telling if the jump along that vector happened for a specific atom in a specific frame
            # 2 - (component) list of the unit vector related to the looked jump at every frame
            for jump_comp, component in jump_and_vector:
                # Go through every jump frame
                for frame in np.flip(which_frame):
                    # Create a boolean vector selecting the atom that jumped from the jump frame onward
                    which_atoms = np.repeat(
                        jump_comp[frame : frame + 1],
                        self.__atoms_posis.shape[0],
                        axis=0,
                    )

                    idx, _, _ = np.indices(which_atoms.shape)
                    which_atoms = np.where(idx > frame, which_atoms, False)

                    # Subtract or add the jump vector only to the jump atoms in all the frame after the jump
                    self.__atoms_posis = np.where(  # pyright: ignore
                        which_atoms,
                        self.__atoms_posis + mod * component[frame : frame + 1],
                        self.__atoms_posis,
                    )

    def __remove_drift(self) -> None:
        drift = self.__atoms_posis.mean(1)
        drift = drift - drift[0]

        self.__atoms_posis = self.__atoms_posis - drift.reshape(
            drift.shape[0], 1, drift.shape[1]
        )

    def __get_atoms_xdatcar(self, xdatcar_path: str) -> None:
        self.__atoms = dict()

        with open(xdatcar_path, "r") as data:
            for _ in range(5):
                data.readline()

            species = data.readline().strip().split()
            numbers = data.readline().strip().split()

            for key, val in zip(species, numbers):
                self.__atoms[key] = int(val)

    def __get_atoms_string(self) -> str:
        res = "  "
        res += "  ".join(self.__atoms.keys())
        res += "\n  "
        res += "  ".join([str(n) for n in self.__atoms.values()])

        return res + "\n"

    def __is_cell_printed(self, xdatcar_path: str) -> bool:
        n_atoms = sum(self.__atoms.values())

        with open(xdatcar_path, "r") as file:
            for _ in range(n_atoms + 8):
                file.readline()

            return file.readline().strip().split()[0] != "Direct"


if __name__ == "__main__":
    from time import time
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog="Compute MSD",
        description="Simple CLI that read an XDATCAR and evaluate the MSD",
    )

    parser.add_argument(
        "xdatcar_path",
        nargs="?",
        type=str,
        default="XDATCAR",
        help="Path to the XDATCAR file to read from, if not inserted a file named XDATCAR will be searched in the current directory",
    )

    parser.add_argument(
        "-sc",
        "--start_conf",
        type=int,
        default=0,
        help="Number of configuration to skip while reading the XDATCAR",
    )

    parser.add_argument(
        "-p",
        "--potim",
        type=float,
        default=1,
        help="Value of the POTIM variable used in the simulation",
    )

    args = parser.parse_args()

    # REAL APPLICATION
    anal = VaspMDAnalyzer(args.xdatcar_path, args.potim, start_conf=args.start_conf)

    print("\nCompute MSD using FFT:")
    start = time()
    np.save("MSD_FFT", anal.get_MSD("H", "fft"))
    print(f"Finished in {time() - start:.3f}s")

    anal.plot_MSD("H")
    anal.write("XDATCAR_corretto")
