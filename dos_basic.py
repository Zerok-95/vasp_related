
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.outputs import Vasprun, Locpot
from pymatgen.electronic_structure.core import OrbitalType, Spin


class PDOS:
    def __init__(self, dir_path="./"):
        self.dir_path = dir_path
        self.projected = None

    def read_vasprun(self, filename='vasprun.xml', align_efermi=True):
        # pickle 파일이 있으면 로드
        pklfile = os.path.join(self.dir_path, "pdos.pkl")
        if os.path.isfile(pklfile):
            print(f"Found existing {pklfile}. Loading data...")
            with open(pklfile, 'rb') as f:
                self.structure, self.dos = pickle.load(f)

        # 2) 없으면 vasprun.xml 읽어서 계산 후 저장
        else:
            vasprun_file = os.path.join(self.dir_path, filename)
            self.vasprun = Vasprun(vasprun_file)
            self.structure = self.vasprun.final_structure
            self.dos = self.vasprun.complete_dos
            with open(pklfile, 'wb') as f:
                pickle.dump((self.structure, self.dos), f)
            print(f"Saved data as {pklfile}")

        if align_efermi is True:
            self.energies = self.dos.energies - self.dos.efermi
            self.efermi = 0
        else:
            self.energies = self.dos.energies
            self.efermi = self.dos.efermi

        self.densities = self.dos.densities[Spin.up]
        self.selected_indices = range(len(self.structure))

    def site_selective(self, sites, unique=None, cutting=None):
        if isinstance(sites, str):
            selected_indices = [i for i, site in enumerate(self.structure) if site.specie.symbol == sites]
            self.selected_indices = [i for i in self.selected_indices if i in selected_indices]
        elif isinstance(sites, list):
            selected_indices = sites
            self.selected_indices = [i for i in self.selected_indices if i in selected_indices]
        elif sites is None:
            pass
        else:
            raise ValueError("sites have to be str(element) or list(sites)")

        ### Custom ###
        # cutting = [2, 5, 10] [axis, min, max]
        if cutting is not None:
            z_coords = np.array([site.coords[cutting[0]] for site in self.structure])
            z_min, z_max = cutting[1], cutting[2],
            selected_indices = [i for i, site in enumerate(self.structure)
                                if z_min < site.coords[cutting[0]] < z_max]
            self.selected_indices = [i for i in self.selected_indices if i in selected_indices]

        if unique is not None:
            sga = SpacegroupAnalyzer(self.structure, symprec=1e-3)
            sym_struct = sga.get_symmetrized_structure()
            equiv_idx_groups = sym_struct.equivalent_indices
            unique_indices = [group[0] for group in equiv_idx_groups]
            self.selected_indices = [i for i in self.selected_indices if i in unique_indices]

    def projection(self, orbitals=None):
        """
        - orbitals이 auto면 block 값에 대해 반환, ["p", "s"] 지정 가능
        """
        spd_dos_list, symbol_list, orbital_list = [], [], []
        # site indices 에 해당하는 애들 dos 가져오기
        for i,site in enumerate(self.structure):
            # orbital 선택하기
            if i in self.selected_indices:
                # print(i, site)
                if orbitals is None:
                    orbital = "all"
                    spd_dos = self.dos.get_site_dos(site)
                elif orbitals == 'auto':
                    orbital = {"s": OrbitalType.s, "p": OrbitalType.p, "d": OrbitalType.d}[site.specie.block.lower()]
                    spd_dos = self.dos.get_site_spd_dos(site)[orbital]
                else:
                    orbital = {"s": OrbitalType.s, "p": OrbitalType.p, "d": OrbitalType.d}[orbitals[i]]
                    spd_dos = self.dos.get_site_spd_dos(site)[orbital]

                spd_dos_list.append(spd_dos.densities[Spin.up])
                symbol_list.append(site.specie.symbol)
                orbital_list.append(orbital)

        self.densities = np.array(spd_dos_list)
        self.energies = np.repeat(self.energies[np.newaxis, :], len(self.selected_indices), axis=0)
        self.projected = True
        self.symbols, self.orbitals = symbol_list, orbital_list
        print(f"Projected DOS for {len(self.selected_indices)} sites")

    def only_valence(self):
        mask_cond = self.energies > self.efermi
        self.densities[mask_cond] = 0

    def apply_vacuum_level(self, axis):
        # LOCPOT 파일 읽기
        locpot_path = os.path.join(self.dir_path, "LOCPOT")
        locpot = Locpot.from_file(locpot_path)

        # planar-average 잠재력 얻고 최대값(vacuum level) 선택
        planar_avg = locpot.get_average_along_axis(axis)
        self.vacuum_level = max(planar_avg)
        ref_energy = self.vacuum_level

        # 3) DOS 에너지와 페르미레벨 조정
        self.energies = self.dos.energies - ref_energy
        self.efermi = self.dos.efermi - ref_energy

        print("Energies have been shifted by electrostatic potential at core")

    def interpolate_dos(self, emin, emax, n):
        """
        에너지 범위 [emin, emax]에서 n개의 grid 포인트로 DOS 데이터를 선형 보간합니다.

        결과는 self.interpolated_energy, self.interpolated_density에 저장됩니다.
        """
        interp_energy = np.linspace(emin, emax, n)
        interp_density = []

        if self.projected:
            for eng, dens in zip(self.energies, self.densities):
                interp_dens = np.interp(interp_energy, eng, dens)
                interp_density.append(interp_dens)
            self.densities = np.array(interp_density)
            self.energies = np.repeat(interp_energy[np.newaxis, :], len(self.selected_indices), axis=0)
        else:
            interp_dens = np.interp(interp_energy, self.energies, self.densities)
            self.densities = interp_dens
            self.energies = interp_energy


    def gaussian_filter(self, sigma=0.5):
        """
        보간된 DOS 데이터에 대해 가우시안 필터링을 수행합니다.
        기본 sigma는 0.5입니다.

        필터링 결과는 self.filtered_density에 저장됩니다.
        """

        gauss_density = []

        if self.projected:
            for dens in self.densities:
                gauss_dens = gaussian_filter1d(dens, sigma)
                gauss_density.append(gauss_dens)
            self.densities = np.array(gauss_density)
        else:
            gauss_dens = gaussian_filter1d(self.densities, sigma)
            self.densities = gauss_dens

    def plot(self, mode="all", emin=None, emax=None):
        """
        matplotlib을 사용하여 DOS 데이터를 플롯합니다.
        보간된 데이터가 있고, 필터링된 데이터가 있다면 필터링된 데이터를 사용합니다.

        Parameters:
          mode : str, 옵션 "all" 또는 "separated"
                 "all": 모든 DOS curve를 하나의 그래프에 출력합니다.
                 "separated": 각 DOS curve를 개별 subplot에 수직으로 나열합니다.
        """

        emin =  emin if emin is not None else np.min(self.energies)
        emax =  emax if emax is not None else np.max(self.energies)
        mask = (self.energies >= emin) & (self.energies <= emax)
        if self.projected is True:
            n_curves = len(self.densities)
            print(f"# of sites is {n_curves}.")
            ymax = np.max(self.densities[mask])
        else:
            ymax = np.max(self.densities[mask])

        if mode == "all":
            plt.figure(figsize=(8,4))
            if self.projected is True:
                alpha = max(0.1, 1.0 - 0.3 * np.log10(n_curves + 1))
                for i in range(n_curves):
                    label = f"Site {self.selected_indices[i] + 1}: {self.symbols[i]}-{self.orbitals[i]}"
                    plt.plot(self.energies[i], self.densities[i], label=label, alpha=alpha)
                print(f"alpha: {alpha}")
                if n_curves <= 10:
                    plt.legend(fontsize=12)
            else:
                plt.plot(self.energies, self.densities)

            plt.xlabel("Energy (eV)", fontsize=18)
            plt.ylabel("DOS (states/eV)", fontsize=18)

            plt.xlim(emin, emax)
            plt.ylim(0, ymax*1.15)
            plt.tight_layout()

        elif mode == "separated":
            if n_curves == 1:
                raise ValueError("Need more than 1 site")

            # 파스텔 톤 색상 지정
            cmap = plt.get_cmap("Pastel1")
            colors = cmap(np.linspace(0, 1, n_curves))

            fig, axes = plt.subplots(n_curves, 1, figsize=(6, 1.3 * n_curves), sharex=True, gridspec_kw={'hspace': 0.12})
            for i, ax in enumerate(axes):
                x, y = self.energies[i], self.densities[i]
                color = colors[i]
                dark_color = tuple(max(0, c * 0.5) for c in color[:3])

                # 채우기(filled curve)
                ax.fill_between(x, y, color=color, alpha=0.8)
                # 경계선(plot)
                ax.plot(x, y, color=dark_color, linewidth=1.5)
                ax.set_xlim(emin, emax)
                # 해당 에너지 구간 내 y 최대값 계산 후 약간 여유 두고 설정
                mask = (x >= emin) & (x <= emax)
                max_y = y[mask].max() if mask.any() else y.max()
                ax.set_ylim(0, max_y * 1.15)
                # 중앙 축에만 y축 레이블
                if i == n_curves // 2:
                    ax.set_ylabel("DOS (states/eV)", fontsize=18)

                # 눈금 모두 안쪽으로
                ax.tick_params(
                    axis='both',
                    which='both',
                    direction='in',
                    top=True,  # 위쪽 눈금도 켜기
                    right=True,  # 오른쪽 눈금은 끄기
                    labelbottom=(i == n_curves - 1)  # 맨 아래만 x축 레이블
                )

                # 틱 라벨 크기
                ax.tick_params(labelsize=12)

            # 맨 아래 축에만 x축 레이블
            axes[-1].set_xlabel("Energy (eV)", fontsize=18)

            plt.tight_layout()
        plt.show()

    # It hasn't been completed
    def save_dat(self, filename="pdos.dat"):
        with open(filename, "w") as f:
            f.write("# Energy (eV)    DOS (states/eV)\n")
            for E, dos in zip(self.energies, self.densities):
                f.write(f"{E:.6f}    {dos:.6f}\n")


if __name__ == "__main__":
    pdos = PDOS()
    pdos.read_vasprun(align_efermi=True)                             # read data from vasprun
    pdos.site_selective("O")          # elements
    # pdos.site_selective([52,54])          # site selective
    # pdos.site_selective("O",cutting=[2,8,12])      # cutting=[axis,min,max]

    pdos.projection(orbitals="auto")
    # pdos.only_valence()
    # pdos.interpolate_dos(emin=-8, emax=1, n=900)       # interpolation
    pdos.gaussian_filter(sigma=4)                    # guassian brodening

    pdos.plot(mode="all",emin=-7, emax=2)             # mode : "all" or "separated"
