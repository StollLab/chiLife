from .RotamerLibrary import RotamerEnsemble


class dRotamerEnsemble(RotamerEnsemble):
    def __init__(self, label, site, increment, protein=None, chain=None, **kwargs):
        """ """
        self.label = label
        self.res = label
        self.site = site
        self.site2 = site + increment
        self.increment = increment
        self.kwargs = kwargs

        self.protein = protein
        self.chain = chain if chain is not None else self.guess_chain()
        self.protein_tree = self.kwargs.setdefault("protein_tree", None)

        self.name = self.res
        if self.site is not None:
            self.name = f"{self.site}_{self.site2}_{self.res}"
        if self.chain is not None:
            self.name += f"_{self.chain}"

        self.selstr = (
            f"resid {self.site} {self.site2} and segid {self.chain} and not altloc B"
        )

        self.forgive = kwargs.setdefault("forgive", 1.0)
        self.clash_radius = kwargs.setdefault("clash_radius", 14.0)
        self._clash_ori_inp = kwargs.setdefault("clash_ori", "cen")
        self.superimposition_method = kwargs.setdefault(
            "superimposition_method", "bisect"
        ).lower()
        self.dihedral_sigma = kwargs.setdefault("dihedral_sigma", 25)
        self.minimize = kwargs.pop("minimize", True)
        self.eval_clash = kwargs["eval_clash"] = False

        self.get_lib()
        self.protein_setup()
        self.sub_labels = (self.SL1, self.SL2)

    def protein_setup(self):
        self.protein = self.protein.select_atoms("not (byres name OH2 or resname HOH)")
        self.clash_ignore_idx = self.protein.select_atoms(
            f"resid {self.site} and segid {self.chain}"
        ).ix

        self.resindex = self.protein.select_atoms(self.selstr).residues[0].resindex
        self.segindex = self.protein.select_atoms(self.selstr).residues[0].segindex

        if self.protein_tree is None:
            self.protein_tree = cKDTree(self.protein.atoms.positions)

        if self.minimize:
            self._minimize()

    def guess_chain(self):
        if self.protein is None:
            chain = "A"
        elif len(nr_segids := set(self.protein.segments.segids)) == 1:
            chain = self.protein.segments.segids[0]
        elif np.isin(self.protein.residues.resnums, self.site).sum() == 0:
            raise ValueError(
                f"Residue {self.site} is not present on the provided protein"
            )
        elif np.isin(self.protein.residues.resnums, self.site).sum() == 1:
            chain = self.protein.select_atoms(f"resid {self.site}").segids[0]
        else:
            raise ValueError(
                f"Residue {self.site} is present on more than one chain. Please specify the desired chain"
            )
        return chain

    def get_lib(self):
        PhiSel, PsiSel = None, None
        if self.protein is not None:
            # get site backbone information from protein structure
            sel_txt = f"resnum {self.site} and segid {self.chain}"
            PhiSel = self.protein.select_atoms(sel_txt).residues[0].phi_selection()
            PsiSel = self.protein.select_atoms(sel_txt).residues[0].psi_selection()

        # Default to helix backbone if none provided
        Phi = (
            None
            if PhiSel is None
            else np.rad2deg(chiLife.get_dihedral(PhiSel.positions))
        )
        Psi = (
            None
            if PsiSel is None
            else np.rad2deg(chiLife.get_dihedral(PsiSel.positions))
        )

        with open(
                chiLife.RL_DIR / f"residue_internal_coords/{self.label}_ic.pkl", "rb"
        ) as f:
            self.cst_idxs, self.csts = pickle.load(f)

        self.SL1 = chiLife.SpinLabel(
            self.label + "i", self.site, self.protein, self.chain, **self.kwargs
        )
        self.SL2 = chiLife.SpinLabel(
            self.label + f"ip{self.increment}",
            self.site2,
            self.protein,
            self.chain,
            **self.kwargs,
        )

    def save_pdb(self, name=None):
        if name is None:
            name = self.name + ".pdb"
        if not name.endswith(".pdb"):
            name += ".pdb"

        chiLife.save(name, self.SL1, self.SL2)

    def _minimize(self):
        def objective(dihedrals, ic1, ic2, opt):
            coords1 = ic1.set_dihedral(
                dihedrals[: len(self.SL1.dihedral_atoms)], 1, self.SL1.dihedral_atoms
            ).to_cartesian()
            coords2 = ic2.set_dihedral(
                dihedrals[-len(self.SL2.dihedral_atoms) :], 1, self.SL2.dihedral_atoms
            ).to_cartesian()

            distances = np.linalg.norm(
                coords1[self.cst_idxs[:, 0]] - coords2[self.cst_idxs[:, 1]], axis=1
            )
            diff = distances - opt
            return diff @ diff

        for i, (ic1, ic2) in enumerate(
            zip(self.SL1.internal_coords, self.SL2.internal_coords)
        ):
            d0 = np.concatenate(
                [
                    ic1.get_dihedral(1, self.SL1.dihedral_atoms),
                    ic2.get_dihedral(1, self.SL2.dihedral_atoms),
                ]
            )
            lb = [-np.pi] * len(d0)  # d0 - np.deg2rad(40)  #
            ub = [np.pi] * len(d0)  # d0 + np.deg2rad(40) #
            bounds = np.c_[lb, ub]
            xopt = opt.minimize(
                objective, x0=d0, args=(ic1, ic2, self.csts[i]), bounds=bounds
            )
            print(xopt.fun)
            self.SL1._coords[i] = ic1._coords[self.SL1.H_mask]
            self.SL2._coords[i] = ic2._coords[self.SL2.H_mask]

    @property
    def weights(self):
        return self.SL1.weights

    @property
    def coords(self):
        return np.concatenate([self.SL1._coords, self.SL2._coords], axis=1)

    @property
    def spin_coords(self):
        sc_matrix = [
            SL.spin_coords
            for SL in self.sub_labels
            if not np.any(np.isnan(SL.spin_coords))
        ]
        return np.sum(sc_matrix, axis=0) / len(sc_matrix)

    @property
    def spin_centroid(self):
        return np.average(self.spin_coords, weights=self.weights, axis=0)

    @property
    def centroid(self):
        return self.coords.mean(axis=(0, 1))

    @property
    def clash_ori(self):

        if isinstance(self._clash_ori_inp, (np.ndarray, list)):
            if len(self._clash_ori_inp) == 3:
                return self._clash_ori_inp
        elif isinstance(self._clash_ori_inp, str):
            if self._clash_ori_inp in ["cen", "centroid"]:
                return self.centroid()
            elif (ori_name := self._clash_ori_inp.upper()) in self.atom_names:
                return np.squeeze(self.coords[0][ori_name == self.atom_names])
        else:
            raise ValueError(
                f"Unrecognized clash_ori option {self._clash_ori_inp}. Please specify a 3D vector, an "
                f"atom name or `centroid`"
            )

        return self._clash_ori

    @clash_ori.setter
    def clash_ori(self, inp):
        self._clash_ori_inp = inp

    def evaluate(self, environment, environment_tree=None, ignore_idx=None, temp=298):

        if self.protein_tree is None:
            self.protein_tree = cKDTree(self.protein.atoms.positions)
        self.evaluate_clashes(
            self.protein, self.protein_tree, ignore_idx=self.clash_ignore_idx
        )

        self.trim()

    def evaluate_clashes(
        self, environment, environment_tree=None, ignore_idx=None, temp=298
    ):
        """
        Measure lennard-jones clashes of the spin label in a given environment and reweight/trim rotamers of the
        SpinLabel.
        :param environment: MDAnalysis.Universe, MDAnalysis.AtomGroup
            The protein environment to be considered when evaluating clashes.
        :param environment_tree: cKDtree
            k-dimensional tree of atom coordinates of the environment.
        :param ignore_idx: array-like
            list of atom coordinate indices to ignore when evaluating clashes. Usually the native amino acid at the
            SpinLable site.
        :param temp: float
            Temperature to consider when re-weighting rotamers.
        """

        if environment_tree is None:
            environment_tree = cKDTree(environment.positions)

        probabilities = chiLife.evaluate_clashes(
            ori=self.clash_ori,
            label_library=self.coords[:, self.side_chain_idx],
            label_lj_rmin2=self.rmin2,
            label_lj_eps=self.eps,
            environment=environment,
            environment_tree=environment_tree,
            ignore_idx=ignore_idx,
            temp=temp,
            energy_func=self.energy_func,
            clash_radius=self.clash_radius,
            forgive=self.forgive,
        )

        self.weights, self.partition = chiLife.reweight_rotamers(
            probabilities, self.weights, return_partition=True
        )
        logging.info(f"Relative partition function: {self.partition:.3}")
