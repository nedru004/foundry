import json
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyrosetta as pr
from Bio.PDB import PDBParser, Selection
from pyrosetta.rosetta.core.io import pose_from_pose
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import (
    OperateOnResidueSubset,
    PreventRepackingRLT,
    RestrictToRepacking,
)
from pyrosetta.rosetta.core.select import get_residues_from_subset
from pyrosetta.rosetta.core.select import residue_selector as rs
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
from pyrosetta.rosetta.core.simple_metrics.metrics import RMSDMetric
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from pyrosetta.rosetta.protocols.simple_moves import AlignChainMover
from scipy.spatial import cKDTree

from boltz_ph.constants import RESTYPE_3TO1, HYDROPHOBIC_AA
from utils.metrics import get_CA_and_sequence, np_rmsd, radius_of_gyration

# Initialize PyRosetta with all needed options
dalphaball_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "DAlphaBall.gcc"
)
pr.init(
    f"-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {dalphaball_path} -corrections::beta_nov16 true -relax:default_repeats 1"
)

def get_sequence(cif_file, chain_id="B"):
    """
    Extracts 1-letter protein sequence from a CIF file.
    Note: CIF parsing now relies on Bio.PDB.MMCIFParser which is used inside get_CA_and_sequence.
    """
    try:
        _, sequence = get_CA_and_sequence(cif_file, chain_id)
        return sequence
    except Exception as e:
        print(f"Error getting sequence from {cif_file} chain {chain_id}: {e}")
        return ""


def clean_pdb(pdb_file):
    """
    Removes non-standard lines from a PDB file to ensure PyRosetta compatibility.
    """
    with open(pdb_file) as f_in:
        relevant_lines = [
            line
            for line in f_in
            if line.startswith(("ATOM", "HETATM", "MODEL", "TER", "END"))
        ]

    with open(pdb_file, "w") as f_out:
        f_out.writelines(relevant_lines)


def pr_relax(pdb_file, relaxed_pdb_path):
    """
    Runs PyRosetta FastRelax protocol on a PDB file.
    The implementation is robust to handle existence check, PDB loading, and alignment.
    """
    if os.path.exists(relaxed_pdb_path):
        return

    # Generate pose
    try:
        pose = pr.pose_from_pdb(pdb_file)
    except Exception as e:
        print(f"Error loading PDB {pdb_file} for relaxation: {e}")
        return
        
    start_pose = pose.clone()

    ### Generate movemaps
    mmf = MoveMap()
    mmf.set_chi(True)
    mmf.set_bb(True)
    mmf.set_jump(False)

    # Run FastRelax
    fastrelax = FastRelax()
    scorefxn = pr.get_fa_scorefxn()
    fastrelax.set_scorefxn(scorefxn)
    fastrelax.set_movemap(mmf)
    fastrelax.max_iter(200)
    fastrelax.min_type("lbfgs_armijo_nonmonotone")
    fastrelax.constrain_relax_to_start_coords(True)
    fastrelax.apply(pose)

    # Align relaxed structure to original trajectory
    # Uses chain 0 (the whole complex) for alignment
    align = AlignChainMover()
    align.source_chain(0)
    align.target_chain(0)
    align.pose(start_pose)
    align.apply(pose)
    
    # Copy B factors from start_pose to pose for visualization consistency
    for resid in range(1, pose.total_residue() + 1):
        if pose.residue(resid).is_protein():
            # Get the B factor of the first heavy atom in the residue
            bfactor = start_pose.pdb_info().bfactor(resid, 1)
            for atom_id in range(1, pose.residue(resid).natoms() + 1):
                pose.pdb_info().bfactor(resid, atom_id, bfactor)


    # output relaxed and aligned PDB
    pose.dump_pdb(relaxed_pdb_path)
    clean_pdb(relaxed_pdb_path)



def hotspot_residues(trajectory_pdb, binder_chain="B", target_chain="A", atom_distance_cutoff=4.0):
    """
    Identify interface residues on the binder chain by checking which binder atoms
    are within cutoff of ANY atom in ANY non-binder chain.

    Target = all chains except binder_chain.
    """

    # Default 3→1 mapping
    aa3to1_map = {
        "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q",
        "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
        "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"
    }

    parser = PDBParser(QUIET=True)

    # -----------------------
    # Load structure
    # -----------------------
    try:
        structure = parser.get_structure("complex", trajectory_pdb)
    except Exception as e:
        print(f"[ERROR] Could not parse PDB: {e}")
        return {}

    model = structure[0]

    # -----------------------
    # Validate binder chain
    # -----------------------
    if binder_chain not in model:
        print(f"[WARNING] Binder chain '{binder_chain}' not found.")
        return {}

    # -----------------------
    # Build binder atom list
    # -----------------------
    binder_atoms = Selection.unfold_entities(model[binder_chain], "A")
    if len(binder_atoms) == 0:
        print(f"[WARNING] Binder chain '{binder_chain}' has no atoms.")
        return {}

    # -----------------------
    # Build TARGET atom list = all other chains
    # -----------------------
    target_atoms = []
    for chain in model:
        if chain.id != binder_chain:
            target_atoms.extend(Selection.unfold_entities(chain, "A"))

    if len(target_atoms) == 0:
        print("[WARNING] No non-binder chains found for target atoms.")
        return {}

    # -----------------------
    # KD-tree for fast contact search
    # -----------------------
    binder_coords = np.array([a.coord for a in binder_atoms])
    target_coords = np.array([a.coord for a in target_atoms])

    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)

    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    # -----------------------
    # Collect interacting residues
    # -----------------------
    interacting = {}

    for binder_idx, close_list in enumerate(pairs):
        if not close_list:
            continue

        atom = binder_atoms[binder_idx]
        residue = atom.get_parent()

        resnum = residue.id[1]
        res3 = residue.get_resname().upper()
        aa1 = aa3to1_map.get(res3, "X")

        interacting[resnum] = aa1

    return interacting



def score_interface(pdb_file, pdb_file_collapsed, binder_chain="B", target_chain="A"):
    """
    Calculates various PyRosetta interface and complex metrics.
    """
    # Load pose
    try:
        pose = pr.pose_from_pdb(pdb_file_collapsed)
    except Exception as e:
        print(f"Error loading PDB {pdb_file_collapsed} for interface scoring: {e}")
        return {}, {}, ""

    # Define interface string for InterfaceAnalyzerMover
    interface_string = f"{binder_chain}_{target_chain}"

    # analyze interface statistics
    iam = InterfaceAnalyzerMover()
    iam.set_interface(interface_string)
    scorefxn = pr.get_fa_scorefxn()
    iam.set_scorefunction(scorefxn)
    # Enable all calculations
    iam.set_compute_packstat(True)
    iam.set_compute_interface_energy(True)
    iam.set_calc_dSASA(True)
    iam.set_calc_hbond_sasaE(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_separated(True)
    iam.apply(pose)

    # 1. Hotspot Analysis
    interface_residues_set = hotspot_residues(pdb_file, binder_chain, target_chain)
    interface_residues_pdb_ids = []
    interface_AA = dict.fromkeys("ACDEFGHIKLMNPQRSTVWY", 0)

    for pdb_res_num, aa_type in interface_residues_set.items():
        interface_AA[aa_type] = interface_AA.get(aa_type, 0) + 1
        interface_residues_pdb_ids.append(f"{binder_chain}{pdb_res_num}")

    interface_nres = len(interface_residues_pdb_ids)
    interface_residues_pdb_ids_str = ",".join(interface_residues_pdb_ids)

    # 2. Interface Hydrophobicity
    hydrophobic_count = sum(interface_AA[aa] for aa in HYDROPHOBIC_AA)
    interface_hydrophobicity = (
        (hydrophobic_count / interface_nres) * 100 if interface_nres != 0 else 0
    )

    # 3. Retrieve IAM scores
    interfacescore = iam.get_all_data()
    interface_sc = interfacescore.sc_value
    interface_interface_hbonds = interfacescore.interface_hbonds
    interface_dG = iam.get_interface_dG()
    interface_dSASA = iam.get_interface_delta_sasa()
    interface_packstat = iam.get_interface_packstat()
    interface_dG_SASA_ratio = interfacescore.dG_dSASA_ratio * 100
    
    # 4. Buried Unsaturated Hbonds (BUNS)
    buns_filter = XmlObjects.static_get_filter(
        '<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" ignore_surface_res="false" use_ddG_style="true" dalphaball_sasa="1" probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />'
    )
    interface_delta_unsat_hbonds = buns_filter.report_sm(pose)

    # 5. Percentage scores
    interface_hbond_percentage = (
        (interface_interface_hbonds / interface_nres) * 100 if interface_nres != 0 else None
    )
    interface_bunsch_percentage = (
        (interface_delta_unsat_hbonds / interface_nres) * 100 if interface_nres != 0 else None
    )

    # 6. Binder Energy and SASA
    chain_design = ChainSelector(binder_chain)
    tem = pr.rosetta.core.simple_metrics.metrics.TotalEnergyMetric()
    tem.set_scorefunction(scorefxn)
    tem.set_residue_selector(chain_design)
    binder_score = tem.calculate(pose)

    bsasa = pr.rosetta.core.simple_metrics.metrics.SasaMetric()
    bsasa.set_residue_selector(chain_design)
    binder_sasa = bsasa.calculate(pose)

    interface_binder_fraction = (
        (interface_dSASA / binder_sasa) * 100 if binder_sasa > 0 else 0
    )

    # 7. Surface Hydrophobicity (on the binder chain)
    try:
        binder_pose = {
            pose.pdb_info().chain(pose.conformation().chain_begin(i)): p
            for i, p in zip(range(1, pose.num_chains() + 1), pose.split_by_chain())
        }[binder_chain]

        layer_sel = pr.rosetta.core.select.residue_selector.LayerSelector()
        layer_sel.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
        surface_res = layer_sel.apply(binder_pose)

        exp_apol_count = 0
        total_count = 0

        # count apolar and aromatic residues at the surface
        for i in range(1, len(surface_res) + 1):
            if surface_res[i]:
                res = binder_pose.residue(i)
                # Count apolar and aromatic residues as hydrophobic
                if res.is_apolar() or res.name in ["PHE", "TRP", "TYR"]:
                    exp_apol_count += 1
                total_count += 1

        surface_hydrophobicity = exp_apol_count / total_count if total_count > 0 else 0
    except Exception as e:
        print(f"Warning: Failed surface hydrophobicity calculation: {e}")
        surface_hydrophobicity = 0.0

    # 8. Compile and Round Results
    interface_scores = {
        "binder_score": binder_score,
        "surface_hydrophobicity": surface_hydrophobicity,
        "interface_sc": interface_sc,
        "interface_packstat": interface_packstat,
        "interface_dG": interface_dG,
        "interface_dSASA": interface_dSASA,
        "interface_dG_SASA_ratio": interface_dG_SASA_ratio,
        "interface_fraction": interface_binder_fraction,
        "interface_hydrophobicity": interface_hydrophobicity,
        "interface_nres": interface_nres,
        "interface_interface_hbonds": interface_interface_hbonds,
        "interface_hbond_percentage": interface_hbond_percentage,
        "interface_delta_unsat_hbonds": interface_delta_unsat_hbonds,
        "interface_delta_unsat_hbonds_percentage": interface_bunsch_percentage,
    }

    interface_scores = {
        k: round(v, 2) if isinstance(v, float) else v
        for k, v in interface_scores.items()
    }

    return interface_scores, interface_AA, interface_residues_pdb_ids_str


def _get_residue_pairs_within_distance(pdb_file, binder_id, target_id, distance_threshold=10.0):
    """
    Returns set of (binder_resnum, target_resnum) where CA/CB distance <= distance_threshold.
    Used for defining interface residue pairs for per-residue energy.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]
    if binder_id not in model or target_id not in model:
        return set()
    binder_chain = model[binder_id]
    target_chain = model[target_id]
    selected_pairs = set()
    for res1 in binder_chain:
        coord1 = None
        if "CB" in res1:
            coord1 = res1["CB"].coord
        elif "CA" in res1:
            coord1 = res1["CA"].coord
        for res2 in target_chain:
            coord2 = None
            if "CB" in res2:
                coord2 = res2["CB"].coord
            elif "CA" in res2:
                coord2 = res2["CA"].coord
            if coord1 is not None and coord2 is not None:
                dist = np.linalg.norm(np.array(coord1) - np.array(coord2))
                if dist <= distance_threshold:
                    selected_pairs.add((res1.id[1], res2.id[1]))
    return selected_pairs


def _write_protein_only_interface_pdb(input_pdb, output_pdb, keep_chains):
    """
    Write a Rosetta-friendly PDB containing only ATOM records from keep_chains.
    This avoids failures from ligands/metals (e.g. Zn) in per-residue scoring.
    """
    keep_chains = set(keep_chains)
    kept_lines = []
    with open(input_pdb, "r") as f_in:
        for line in f_in:
            if line.startswith("ATOM  ") and line[21] in keep_chains:
                kept_lines.append(line)
    if not kept_lines:
        return False
    with open(output_pdb, "w") as f_out:
        f_out.writelines(kept_lines)
        if not kept_lines[-1].startswith("END"):
            f_out.write("END\n")
    return True


def compute_per_residue_interface_energy(
    pdb_file,
    binder_chain="A",
    target_chain="B",
    distance_threshold=10.0,
    relax=True,
    relax_nbr_distance=20.0,
    relax_repeats=2,
    relax_bb=True,
    output_dir=None,
    output_prefix=None,
    return_partner_map=False,
):
    """
    Loads PDB, optionally relaxes the interface neighborhood, scores with full-atom
    scorefunction, and returns per-binder-residue
    summed interface energy (only favorable pair contributions, negative values).
    Relaxation is recommended for more meaningful energies.
    If output_dir and output_prefix are provided, saves the relaxed PDB to
    output_dir/{output_prefix}_relaxed.pdb.
    Returns: (summed_energy_dict, key_res_with_aa_dict[, partner_map]).
    summed_energy_dict: residue_number_str -> total interface energy (float).
    key_res_with_aa_dict: residue_number_str -> (energy, one_letter_aa) for residues
        that have favorable contribution (used by get_fixed_residues_from_interface_energy).
    """
    target_chains = (
        list(target_chain)
        if isinstance(target_chain, (list, tuple, set))
        else [str(target_chain)]
    )
    scorefxn = pr.get_fa_scorefxn()

    # Build a protein-only interface PDB for Rosetta scoring to avoid failures
    # from ligands/metals (e.g. Zn) in the full structure.
    scoring_pdb = pdb_file
    temp_scoring_pdb = None
    try:
        if output_dir and output_prefix:
            os.makedirs(output_dir, exist_ok=True)
            candidate = os.path.join(output_dir, f"{output_prefix}_protein_only.pdb")
        else:
            fd, candidate = tempfile.mkstemp(prefix="iface_protein_only_", suffix=".pdb")
            os.close(fd)
            temp_scoring_pdb = candidate
        if _write_protein_only_interface_pdb(
            pdb_file, candidate, [binder_chain] + target_chains
        ):
            scoring_pdb = candidate
    except Exception as e:
        print(f"WARNING: Could not generate protein-only PDB for scoring: {e}")

    try:
        pose = pr.pose_from_pdb(scoring_pdb)
    except Exception as e:
        print(f"Error loading PDB for per-residue interface energy: {e}")
        if temp_scoring_pdb and os.path.exists(temp_scoring_pdb):
            os.remove(temp_scoring_pdb)
        if return_partner_map:
            return {}, {}, {}
        return {}, {}

    if relax:
        try:
            # Relax only the neighborhood around binder + target chains to keep this
            # feasible inside optimization cycles.
            binder_sel = rs.ChainSelector(binder_chain)
            target_sels = [rs.ChainSelector(tc) for tc in target_chains]
            focus_sel = binder_sel
            for ts in target_sels:
                focus_sel = rs.OrResidueSelector(focus_sel, ts)

            nbr_sel = rs.NeighborhoodResidueSelector()
            nbr_sel.set_focus_selector(focus_sel)
            nbr_sel.set_distance(float(relax_nbr_distance))
            nbr_sel.set_include_focus_in_subset(True)

            others_sel = rs.NotResidueSelector(nbr_sel)

            tf = TaskFactory()
            tf.push_back(RestrictToRepacking())
            tf.push_back(OperateOnResidueSubset(PreventRepackingRLT(), others_sel))

            mm = MoveMap()
            mm.set_chi(True)
            mm.set_bb(bool(relax_bb))
            mm.set_jump(False)

            relax_mover = FastRelax(scorefxn, int(relax_repeats))
            relax_mover.set_task_factory(tf)
            relax_mover.set_movemap(mm)
            relax_mover.constrain_relax_to_start_coords(True)
            relax_mover.apply(pose)

            if output_dir and output_prefix:
                os.makedirs(output_dir, exist_ok=True)
                relaxed_path = os.path.join(output_dir, f"{output_prefix}_relaxed.pdb")
                pose.dump_pdb(relaxed_path)
                clean_pdb(relaxed_path)
        except Exception as e:
            print(f"WARNING: Interface relax failed, scoring unrelaxed pose: {e}")

    scorefxn(pose)
    pdb_info = pose.pdb_info()
    weights = scorefxn.weights()

    def _residue_contact_distance(pose, i, j):
        """
        Return a representative residue-residue distance for contact constraints.
        Prefers CB-CB distance (more indicative of sidechain contact) and falls back
        to CA-CA when CB is unavailable (e.g. GLY).
        """
        ri = pose.residue(i)
        rj = pose.residue(j)
        ai = "CB" if ri.has("CB") else "CA"
        aj = "CB" if rj.has("CB") else "CA"
        vi = ri.xyz(ai)
        vj = rj.xyz(aj)
        return float((vi - vj).norm())

    # Sum favorable (negative) pair energies across all requested target chains.
    res_sum: dict[str, float] = {}
    partner_map = {}
    for tc in target_chains:
        interface_pairs = _get_residue_pairs_within_distance(
            scoring_pdb, binder_chain, tc, distance_threshold
        )
        if not interface_pairs:
            continue

        for binder_resnum, target_resnum in interface_pairs:
            i = pdb_info.pdb2pose(binder_chain, binder_resnum)
            j = pdb_info.pdb2pose(tc, target_resnum)
            if i == 0 or j == 0:
                continue
            emap = pr.rosetta.core.scoring.EMapVector()
            scorefxn.eval_ci_2b(pose.residue(i), pose.residue(j), pose, emap)
            scorefxn.eval_cd_2b(pose.residue(i), pose.residue(j), pose, emap)
            total = 0.0
            end_enum = int(pr.rosetta.core.scoring.end_of_score_type_enumeration)
            for st_int in range(1, end_enum):
                st = pr.rosetta.core.scoring.ScoreType(st_int)
                w = weights[st]
                if w != 0.0:
                    total += w * emap[st]
            key = str(binder_resnum)
            prev = partner_map.get(key)
            if prev is None or total < prev["pair_energy"]:
                contact_distance = _residue_contact_distance(pose, i, j)
                partner_map[key] = {
                    "target_chain": tc,
                    "target_resnum": int(target_resnum),
                    "pair_energy": float(total),
                    "contact_distance": float(contact_distance),
                }
            if total >= 0:
                continue
            res_sum[key] = res_sum.get(key, 0.0) + total

    if not res_sum:
        if temp_scoring_pdb and os.path.exists(temp_scoring_pdb):
            os.remove(temp_scoring_pdb)
        if return_partner_map:
            return {}, {}, {}
        return {}, {}

    key_res_with_aa = {}
    for resnum_str, energy in res_sum.items():
        pose_idx = pdb_info.pdb2pose(binder_chain, int(resnum_str))
        if pose_idx != 0:
            aa_one = pose.residue(pose_idx).name1()
            key_res_with_aa[resnum_str] = (float(energy), aa_one)
    if return_partner_map:
        if temp_scoring_pdb and os.path.exists(temp_scoring_pdb):
            os.remove(temp_scoring_pdb)
        return res_sum, key_res_with_aa, partner_map
    if temp_scoring_pdb and os.path.exists(temp_scoring_pdb):
        os.remove(temp_scoring_pdb)
    return res_sum, key_res_with_aa


def get_fixed_residues_from_interface_energy(
    pdb_file,
    binder_chain="A",
    target_chain="B",
    energy_threshold=-5.0,
    distance_threshold=10.0,
    relax=True,
    relax_nbr_distance=20.0,
    relax_repeats=2,
    relax_bb=True,
    use_temperature_from_score=False,
    temperature_scale=0.05,
    min_temperature=0.01,
    base_temperature=0.1,
    output_dir=None,
    output_prefix=None,
    return_contact_constraints=False,
    contact_max_distance=6.0,
    contact_force=False,
):
    """
    Computes per-residue interface energy and returns inputs for the next design cycle.

    If output_dir and output_prefix are provided, saves:
      - output_dir/{output_prefix}_relaxed.pdb (relaxed structure)
      - output_dir/{output_prefix}_residue_energy.csv (per-residue energies)

    If use_temperature_from_score is False: returns (fixed_residues_str, None).
    fixed_residues_str is space-separated "A1 A5 A10" for binder residues with
    summed interface energy < energy_threshold (strongly favorable), for use as
    LigandMPNN fixed_positions.

    If use_temperature_from_score is True: returns ( "", temperature_per_residue_dict ).
    temperature_per_residue_dict maps "A1" -> float temperature; more negative
    energy -> lower temperature (more fixed). Formula: temp = max(min_temperature,
    base_temperature + energy * temperature_scale).

    If return_contact_constraints is True, also returns Boltz contact constraints:
    [{"contact": {"token1": [CHAIN_ID, RES_IDX], "token2": [CHAIN_ID, RES_IDX], ...}}, ...]

    Returns:
        (fixed_residues_str, temperature_per_residue_dict or None[, contact_constraints])
    """
    # Allow target_chain to be a single chain ("B") or a collection (["B","C"]).
    # Aggregation is done inside compute_per_residue_interface_energy.
    output = compute_per_residue_interface_energy(
        pdb_file,
        binder_chain,
        target_chain,
        distance_threshold,
        relax=relax,
        relax_nbr_distance=relax_nbr_distance,
        relax_repeats=relax_repeats,
        relax_bb=relax_bb,
        output_dir=output_dir,
        output_prefix=output_prefix,
        return_partner_map=return_contact_constraints,
    )
    if return_contact_constraints:
        res_sum, key_res_with_aa, partner_map = output
    else:
        res_sum, key_res_with_aa = output
    if not res_sum:
        if return_contact_constraints:
            return "", None, []
        return "", None

    if output_dir and output_prefix:
        os.makedirs(output_dir, exist_ok=True)
        energy_path = os.path.join(output_dir, f"{output_prefix}_residue_energy.csv")
        rows = []
        for resnum_str, (energy, aa) in key_res_with_aa.items():
            res_id = f"{binder_chain}{resnum_str}"
            is_fixed = float(energy) < energy_threshold
            rows.append({"residue": res_id, "energy": round(float(energy), 3), "aa": aa, "fixed": is_fixed})
        pd.DataFrame(rows).to_csv(energy_path, index=False)

    if use_temperature_from_score:
        temp_dict = {}
        for resnum_str, energy in res_sum.items():
            if float(energy) >= energy_threshold:
                continue
            temp = max(min_temperature, base_temperature + float(energy) * temperature_scale)
            temp_dict[f"{binder_chain}{resnum_str}"] = round(temp, 4)
        if return_contact_constraints:
            contact_constraints = []
            for resnum_str, (energy, _) in key_res_with_aa.items():
                if float(energy) >= energy_threshold:
                    continue
                partner = partner_map.get(str(resnum_str))
                if partner is None:
                    continue
                max_distance = partner.get("contact_distance", contact_max_distance)
                contact_constraints.append(
                    {
                        "contact": {
                            "token1": [binder_chain, int(resnum_str)],
                            "token2": [
                                partner["target_chain"],
                                int(partner["target_resnum"]),
                            ],
                            "max_distance": float(max_distance),
                            "force": bool(contact_force),
                        }
                    }
                )
            return "", temp_dict, contact_constraints
        return "", temp_dict

    fixed = [
        f"{binder_chain}{resnum_str}"
        for resnum_str, (energy, _) in key_res_with_aa.items()
        if float(energy) < energy_threshold
    ]
    fixed_residues_str = " ".join(
        sorted(fixed, key=lambda x: int(x[len(binder_chain):]) if x[len(binder_chain):].isdigit() else 0)
    )
    if return_contact_constraints:
        contact_constraints = []
        for resnum_str, (energy, _) in key_res_with_aa.items():
            if float(energy) >= energy_threshold:
                continue
            partner = partner_map.get(str(resnum_str))
            if partner is None:
                continue
            max_distance = partner.get("contact_distance", contact_max_distance)
            contact_constraints.append(
                {
                    "contact": {
                        "token1": [binder_chain, int(resnum_str)],
                        "token2": [partner["target_chain"], int(partner["target_resnum"])],
                        "max_distance": float(max_distance),
                        "force": bool(contact_force),
                    }
                }
            )
        return fixed_residues_str, None, contact_constraints
    return fixed_residues_str, None


# Removed align_pdbs (was unused or for debugging external files)

def unaligned_rmsd(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    """
    Calculates RMSD between two chains without performing a Kabsch alignment.
    (This function is less commonly used than np_rmsd but kept for completeness).
    """
    reference_pose = pr.pose_from_pdb(reference_pdb)
    align_pose = pr.pose_from_pdb(align_pdb)

    reference_chain_selector = ChainSelector(reference_chain_id)
    align_chain_selector = ChainSelector(align_chain_id)

    reference_chain_subset = reference_chain_selector.apply(reference_pose)
    align_chain_subset = align_chain_selector.apply(align_pose)

    reference_residue_indices = get_residues_from_subset(reference_chain_subset)
    align_residue_indices = get_residues_from_subset(align_chain_subset)
    
    if len(reference_residue_indices) != len(align_residue_indices):
        print("Warning: Chains have different lengths for RMSD calculation. Aligning will fail or be meaningless.")
        return None

    reference_chain_pose = pr.Pose()
    align_chain_pose = pr.Pose()

    pose_from_pose(reference_chain_pose, reference_pose, reference_residue_indices)
    pose_from_pose(align_chain_pose, align_pose, align_residue_indices)

    rmsd_metric = RMSDMetric()
    rmsd_metric.set_comparison_pose(reference_chain_pose)
    rmsd = rmsd_metric.calculate(align_chain_pose)

    return round(rmsd, 2)

def collapse_multiple_chains(pdb_in, pdb_out, binder_chain="A", collapse_target="B"):
    """
    Generalized chain collapse:
      - binder_chain stays as-is (e.g. A)
      - all other chains collapse to collapse_target (e.g. B)
      - Only *one* TER after binder_chain → collapsed-group
      - One final TER at the end of collapsed chains
    """

    with open(pdb_in, "r") as f:
        lines = f.readlines()

    # -------------------------------
    # STEP 1 — Collect ATOM/HETATM lines & detect chains
    # -------------------------------
    atom_indices = []
    chain_list = []
    for i, line in enumerate(lines):
        if line.startswith(("ATOM  ", "HETATM")):
            atom_indices.append(i)
            chain_list.append(line[21])

    # Identify all chains
    all_chains = sorted(set(chain_list))

    # All non-binder chains collapse
    collapse_chains = [c for c in all_chains if c != binder_chain]

    # -------------------------------
    # STEP 2 — Detect chain transitions
    # -------------------------------
    transitions = []
    for (idx1, c1), (idx2, c2) in zip(
            zip(atom_indices, chain_list),
            zip(atom_indices[1:], chain_list[1:])):
        if c1 != c2:
            transitions.append((idx1, c1, c2))

    # -------------------------------
    # STEP 3 — Choose which transitions get TER
    # -------------------------------
    ter_after = set()
    seen_binder_to_collapsed = False

    for idx, c1, c2 in transitions:

        # If binder → collapsed-group transition
        if c1 == binder_chain and c2 in collapse_chains:
            if not seen_binder_to_collapsed:
                ter_after.add(idx)
                seen_binder_to_collapsed = True
            continue

        # All other TERs removed (do nothing)

    # -------------------------------
    # STEP 4 — Add a final TER at end of collapsed chains
    # -------------------------------
    last_collapsed_idx = None
    for i, line in enumerate(lines):
        if line.startswith(("ATOM  ", "HETATM")) and line[21] in collapse_chains:
            last_collapsed_idx = i

    if last_collapsed_idx is not None:
        ter_after.add(last_collapsed_idx)

    # -------------------------------
    # STEP 5 — Write output without old TERs
    # -------------------------------
    temp_out = []
    for i, line in enumerate(lines):

        if line.startswith(("ATOM  ", "HETATM")):
            temp_out.append(line)
            if i in ter_after:
                temp_out.append("TER\n")
            continue

        # remove existing TER lines
        if line.startswith("TER"):
            continue

        temp_out.append(line)

    # -------------------------------
    # STEP 6 — Collapse all other chains into collapse_target
    # -------------------------------
    final_out = []
    for line in temp_out:
        if (line.startswith(("ATOM  ", "HETATM"))
            and line[21] in collapse_chains):
            line = line[:21] + collapse_target + line[22:]

        final_out.append(line)

    with open(pdb_out, "w") as f:
        f.writelines(final_out)

def measure_rosetta_energy(
    pdbs_path,
    pdbs_apo_path,
    save_dir,
    binder_holo_chain="A",
    binder_apo_chain="A",
    target="peptide",
):
    """
    Measures Rosetta energy metrics for a set of designs and filters them based on thresholds.
    """
    # Create relaxed output directory
    relaxed_dir = pdbs_path + "_relaxed"
    os.makedirs(relaxed_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(pdbs_path, "rosetta_energy.csv")

    df = pd.DataFrame()
    processed_files = set()
    
    # Load existing CSV if it exists
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            processed_files = set(existing_df["Model"].values)
            df = existing_df.copy() # Start with existing data
        except Exception as e:
            print(f"Warning: Could not load existing CSV at {output_path}: {e}")

    new_rows = []
    
    for pdb_file in os.listdir(pdbs_path):
        if pdb_file.endswith(".pdb") and not pdb_file.startswith("relax_"):
            if pdb_file in processed_files:
                continue
            try:
                design_pathway = os.path.join(pdbs_path, pdb_file)
                relax_pathway = os.path.join(relaxed_dir, f"relax_{pdb_file}")
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure("protein", design_pathway)
                total_chains = [chain.id for model in structure for chain in model]
                pr_relax(design_pathway, relax_pathway)
                if len(total_chains) > 3:
                    relax_pathway_collapsed = os.path.join(relaxed_dir, f"relax_collapsed_{pdb_file}")
                    collapse_multiple_chains(relax_pathway, relax_pathway_collapsed, binder_chain=binder_holo_chain)
                else:
                    relax_pathway_collapsed = relax_pathway

                (
                    trajectory_interface_scores,
                    trajectory_interface_AA,
                    trajectory_interface_residues,
                ) = score_interface(relax_pathway, relax_pathway_collapsed, binder_chain=binder_holo_chain, target_chain="B")


                print(f"Rosetta scores for {pdb_file}: {trajectory_interface_scores}")

                row_data = {"PDB": relaxed_dir, "Model": f"relax_{pdb_file}"}
                row_data.update(trajectory_interface_scores)
                new_rows.append(row_data)
                processed_files.add(pdb_file)
                    
            except Exception as e:
                print(f"Error processing {pdb_file}: {e}")

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_csv(output_path, index=False)
        print(f"✅ New Rosetta results appended to {output_path}")
    else:
        print("No new PDB files to process for Rosetta scoring.")

    # --- Filtering Logic ---
    if df.empty:
        print("No data available for filtering.")
        return

    # Define filtering mask based on target type and capture thresholds for failure reason
    def failure_reasons(row, target="peptide"):
        reasons = []
        if target == "peptide":
            if not row.get("binder_score", float('inf')) < 0:
                reasons.append("binder_score >= 0")
            if not row.get("surface_hydrophobicity", float('inf')) < 0.35:
                reasons.append("surface_hydrophobicity >= 0.35")
            if not row.get("interface_sc", float('-inf')) > 0.55:
                reasons.append("interface_sc <= 0.55")
            if not row.get("interface_packstat", float('-inf')) > 0:
                reasons.append("interface_packstat <= 0")
            if not row.get("interface_dG", float('inf')) < 0:
                reasons.append("interface_dG >= 0")
            if not row.get("interface_dSASA", float('-inf')) > 1:
                reasons.append("interface_dSASA <= 1")
            if not row.get("interface_dG_SASA_ratio", float('inf')) < 0:
                reasons.append("interface_dG_SASA_ratio >= 0")
            if not row.get("interface_nres", float('-inf')) > 4:
                reasons.append("interface_nres <= 4")
            if not row.get("interface_interface_hbonds", float('-inf')) > 3:
                reasons.append("interface_interface_hbonds <= 3")
            if not row.get("interface_hbond_percentage", float('-inf')) > 0:
                reasons.append("interface_hbond_percentage <= 0")
            if not row.get("interface_delta_unsat_hbonds", float('inf')) < 2:
                reasons.append("interface_delta_unsat_hbonds >= 2")
        else:  # protein, small_molecule, nucleic
            if not row.get("binder_score", float('inf')) < 0:
                reasons.append("binder_score >= 0")
            if not row.get("surface_hydrophobicity", float('inf')) < 0.35:
                reasons.append("surface_hydrophobicity >= 0.35")
            if not row.get("interface_sc", float('-inf')) > 0.55:
                reasons.append("interface_sc <= 0.55")
            if not row.get("interface_packstat", float('-inf')) > 0:
                reasons.append("interface_packstat <= 0")
            if not row.get("interface_dG", float('inf')) < 0:
                reasons.append("interface_dG >= 0")
            if not row.get("interface_dSASA", float('-inf')) > 1:
                reasons.append("interface_dSASA <= 1")
            if not row.get("interface_dG_SASA_ratio", float('inf')) < 0:
                reasons.append("interface_dG_SASA_ratio >= 0")
            if not row.get("interface_nres", float('-inf')) > 7:
                reasons.append("interface_nres <= 7")
            if not row.get("interface_interface_hbonds", float('-inf')) > 3:
                reasons.append("interface_interface_hbonds <= 3")
            if not row.get("interface_hbond_percentage", float('-inf')) > 0:
                reasons.append("interface_hbond_percentage <= 0")
            if not row.get("interface_delta_unsat_hbonds", float('inf')) < 4:
                reasons.append("interface_delta_unsat_hbonds >= 4")
        return '; '.join(reasons) if reasons else None

    if target == "peptide":
        mask = (
            (df["binder_score"] < 0)
            & (df["surface_hydrophobicity"] < 0.35)
            & (df["interface_sc"] > 0.55)
            & (df["interface_packstat"] > 0)
            & (df["interface_dG"] < 0)
            & (df["interface_dSASA"] > 1)
            & (df["interface_dG_SASA_ratio"] < 0)
            & (df["interface_nres"] > 4)
            & (df["interface_interface_hbonds"] > 3)
            & (df["interface_hbond_percentage"] > 0)
            & (df["interface_delta_unsat_hbonds"] < 2)
        )
    else:  # protein, small_molecule, nucleic (uses general protein interface criteria)
        mask = (
            (df["binder_score"] < 0)
            & (df["surface_hydrophobicity"] < 0.35)
            & (df["interface_sc"] > 0.55)
            & (df["interface_packstat"] > 0)
            & (df["interface_dG"] < 0)
            & (df["interface_dSASA"] > 1)
            & (df["interface_dG_SASA_ratio"] < 0)
            & (df["interface_nres"] > 7)
            & (df["interface_interface_hbonds"] > 3)
            & (df["interface_hbond_percentage"] > 0)
            & (df["interface_delta_unsat_hbonds"] < 4)
        )

    # Apply mask and filter
    filtered_df = df[mask].copy()
    failed_df = df[~mask].copy()

    print(f"Number of designs passing all Rosetta filters: {len(filtered_df)}")
    print(f"Number of designs failing Rosetta filters: {len(failed_df)}")

    all_filtered_rows = []
    all_failed_rows = []
    success_sample_num = 0

    def get_metrics(row, pdbs_path, pdbs_apo_path, binder_holo_chain, binder_apo_chain):
        # Returns a new dict with extra annotations (aa_seq, rg, etc)
        row = row.copy()
        try:
            cif_path = row['PDB'] + '/' + row['Model']
            rg, length = radius_of_gyration(cif_path, chain_id=binder_holo_chain)
            model_base = row['Model'].split('relax_')[-1].split('_model.pdb')[0] if row['Model'].startswith('relax') else row['Model'].split('_model.pdb')[0]
            base_path = '/'.join(row['PDB'].split('/')[:-1]) + '/02_design_final_af3/' + model_base
            confidenece_json_1 = f"{base_path}/{model_base}_summary_confidences.json"
            confidenece_json_2 = f"{base_path}/{model_base}_confidences.json"
            af_cif = f"{base_path}/{model_base}_model.cif"
            aa_seq = get_sequence(af_cif, chain_id=binder_holo_chain)
            if row['Model'].startswith('relax'):
                af_holo_pdb = pdbs_path + '/' + row['Model'].split('relax_')[1]
                af_apo_pdb = pdbs_apo_path + '/' + row['Model'].split('relax_')[1]
            else:
                af_holo_pdb = pdbs_path + '/' + row['Model']
                af_apo_pdb = pdbs_apo_path + '/' + row['Model']
            xyz_holo, seq_holo = get_CA_and_sequence(af_holo_pdb, chain_id=binder_holo_chain)
            xyz_apo, seq_apo = get_CA_and_sequence(af_apo_pdb, chain_id=binder_apo_chain)
            rmsd = np_rmsd(xyz_holo, xyz_apo)
            row['apo_holo_rmsd'] = rmsd
            # confidence 1
            try:
                with open(confidenece_json_1, 'r') as f:
                    confidence_data = json.load(f)
                    row['iptm'] = confidence_data['iptm']
            except Exception:
                row['iptm'] = None
            # confidence 2
            try:
                with open(confidenece_json_2, 'r') as f:
                    confidence_data = json.load(f)
                    row['plddt'] = np.mean(confidence_data['atom_plddts'])
                    pae_matrix = np.array(confidence_data['pae'])
                    protein_len = len(aa_seq)
                    interface_pae1 = np.mean(pae_matrix[:protein_len, protein_len:])
                    interface_pae2 = np.mean(pae_matrix[protein_len:, :protein_len])
                    i_pae = (interface_pae1 + interface_pae2) / 2
                    row['i_pae'] = i_pae
            except Exception:
                row['plddt'] = None
                row['i_pae'] = None
            row['rg'] = rg
            row['aa_seq'] = aa_seq
        except Exception as e:
            print(f"Error adding metrics for {row.get('Model', 'unknown')}: {e}")
            row['apo_holo_rmsd'] = None
            row['iptm'] = None
            row['plddt'] = None
            row['i_pae'] = None
            row['rg'] = None
            row['aa_seq'] = None
        return row

    # Process passing (filtered) designs
    if len(filtered_df) > 0:
        for i in range(len(filtered_df)):
            try:
                row = filtered_df.iloc[i].copy()  # Create a copy to avoid SettingWithCopyWarning
                metrics_row = get_metrics(
                    row, pdbs_path, pdbs_apo_path, binder_holo_chain, binder_apo_chain)
                print(
                    f"iptm: {metrics_row.get('iptm', float('nan')):.2f}, "
                    f"plddt: {metrics_row.get('plddt', float('nan')):.1f}, "
                    f"rg: {metrics_row.get('rg', float('nan')):.1f}, "
                    f"i_pae: {metrics_row.get('i_pae', float('nan')):.1f}, "
                    f"apo_holo_rmsd: {metrics_row.get('apo_holo_rmsd', float('nan')):.1f}"
                )
                iptm_val = metrics_row.get('iptm', 0 if metrics_row.get('iptm') is None else metrics_row.get('iptm'))
                plddt_val = metrics_row.get('plddt', 0 if metrics_row.get('plddt') is None else metrics_row.get('plddt'))
                rg_val = metrics_row.get('rg', 999 if metrics_row.get('rg') is None else metrics_row.get('rg'))
                i_pae_val = metrics_row.get('i_pae', 999 if metrics_row.get('i_pae') is None else metrics_row.get('i_pae'))
                rmsd_val = metrics_row.get('apo_holo_rmsd', 999 if metrics_row.get('apo_holo_rmsd') is None else metrics_row.get('apo_holo_rmsd'))
                if iptm_val > 0.5 and plddt_val > 80 and rg_val < 17 and i_pae_val < 15 and rmsd_val < 3.5:
                    shutil.copy(Path(metrics_row['PDB']) / metrics_row['Model'], save_dir + '/' + metrics_row['Model'])
                    all_filtered_rows.append(metrics_row)
                    success_sample_num += 1
                else:
                    fail_row = metrics_row.copy()
                    fail_row['failure_reason'] = (
                        "Does not pass iptm/plddt/rg/i_pae/rmsd thresholds: "
                        f"iptm={iptm_val}, plddt={plddt_val}, rg={rg_val}, i_pae={i_pae_val}, rmsd={rmsd_val}"
                    )
                    all_failed_rows.append(fail_row)
            except Exception as e:
                print(f"Error processing {row['Model']}: {e}")
                continue

    # Process failing designs: also annotate with metrics (aa_seq, rg, etc) and add explicit failure reason(s)
    for i in range(len(failed_df)):
        row = failed_df.iloc[i].copy()
        metrics_row = get_metrics(
            row, pdbs_path, pdbs_apo_path, binder_holo_chain, binder_apo_chain
        )
        metrics_row['failure_reason'] = failure_reasons(row, target=target)
        all_failed_rows.append(metrics_row)

    print("success_sample_num", success_sample_num)

    success_csv = os.path.join(save_dir, 'success_designs.csv')
    failed_csv = os.path.join(save_dir, 'failed_designs.csv')
    zip_path = save_dir + '.zip'

    save_df = pd.DataFrame(all_filtered_rows)
    save_df.to_csv(success_csv, index=False)

    print("Number of Success designs", len(save_df))

    failed_df_save = pd.DataFrame(all_failed_rows)
    failed_df_save.to_csv(failed_csv, index=False)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, save_dir)
                zipf.write(file_path, arcname)


def run_rosetta_step(
    ligandmpnn_dir, af_pdb_dir, af_pdb_dir_apo, binder_id="A", target_type="protein"
):
    """Run Rosetta energy calculation (protein targets only)"""

    if target_type not in ["protein", "peptide"]:
        print("Skipping Rosetta step (not a protein/peptide target)")
        return

    print("Starting Rosetta energy calculation...")
    af_pdb_rosetta_success_dir = f"{ligandmpnn_dir}/af_pdb_rosetta_success"

    measure_rosetta_energy(
        af_pdb_dir,
        af_pdb_dir_apo,
        af_pdb_rosetta_success_dir,
        binder_holo_chain=binder_id,
        binder_apo_chain="A",
        target=target_type,
    )

    print("Rosetta energy calculation completed!")