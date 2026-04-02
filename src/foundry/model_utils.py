import gc
import os
import random
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import gemmi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py3Dmol
import torch
from prody import parsePDB

from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.mol import load_molecules, load_canonicals
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.tokenize.boltz import BoltzTokenizer
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import (
    MSA,
    Coords,
    Ensemble,
    Input,
    Structure,
    StructureV2,
)
from boltz.data.write.pdb import to_pdb
from boltz.main import (
    Boltz2DiffusionParams,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgs,
    PairformerArgsV2,
)
from boltz.model.models.boltz1 import Boltz1
from boltz.model.models.boltz2 import Boltz2

warnings.filterwarnings("ignore", message=".*requires_grad=True.*")
warnings.filterwarnings(
    "ignore",
    message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.*",
    category=UserWarning,
)

alphabet = list[str]("XXARNDCQEGHILKMFPSTWYV-")


def get_CA_and_sequence(pdb_path: str, chain_id: str) -> tuple[np.ndarray, str]:
    """Return CA coordinates and one-letter sequence for a ProDy chain."""
    structure = parsePDB(pdb_path)
    if structure is None:
        raise ValueError(f"Could not parse PDB/CIF path: {pdb_path}")
    chain = structure[chain_id]
    ca = chain.ca
    return ca.getCoords(), str(chain.getSequence())


def chain_sequences_from_atom_array(atom_array) -> dict[str, str]:
    """One-letter protein sequence per chain (standard residues only; skips e.g. waters)."""
    from biotite.structure import get_residue_starts
    from biotite.sequence import ProteinSequence

    sequences: dict[str, str] = {}
    for cid in np.unique(atom_array.chain_id):
        mask = atom_array.chain_id == cid
        sub = atom_array[mask]
        starts = get_residue_starts(sub)
        try:
            seq = "".join(
                ProteinSequence.convert_letter_3to1(sub.res_name[starts])
            )
        except (KeyError, TypeError, ValueError):
            continue
        if seq:
            sequences[str(cid)] = seq
    return sequences


def build_protein_boltz_yaml(
    chain_sequences: dict[str, str],
    msa_paths: dict[str, str | Path],
    *,
    version: int = 1,
    designed_chains_empty_msa: frozenset[str] | None = None,
) -> dict:
    """Build a Boltz/Boltz2 `sequences` YAML dict (protein-only).

    `msa_paths` maps chain id -> path to preprocessed MSA (``.npz`` from
    :func:`process_msa`) or the string ``empty`` for single-sequence mode.
    Chains in ``designed_chains_empty_msa`` force ``msa: empty`` regardless.
    """
    sequences: list = []
    emptyish = designed_chains_empty_msa or frozenset()
    for cid in sorted(chain_sequences.keys(), key=lambda x: (len(x), x)):
        seq = chain_sequences[cid]
        if cid in emptyish:
            msa: str | Path = "empty"
        else:
            msa = msa_paths.get(cid, "empty")
        prot: dict = {"id": cid, "sequence": seq, "msa": str(msa)}
        sequences.append({"protein": prot})
    return {"version": version, "sequences": sequences}


def default_boltz_predict_args(
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    max_parallel_samples: int = 1,
) -> dict:
    return {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "max_parallel_samples": max_parallel_samples,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }


def batch_to_device(batch: dict, device: str | torch.device) -> dict:
    """Move featurized tensors to ``device`` with leading batch dimension (dim 0)."""
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.unsqueeze(0).to(device)
        else:
            out[key] = value
    return out


def boltz_prediction_to_atom_array(structure, output: dict):
    """Write Boltz prediction to a temp PDB and load a Biotite atom array."""
    import tempfile

    from biotite.structure import io

    if output.get("exception"):
        raise RuntimeError("Boltz predict_step failed (e.g. OOM).")

    coords = output["coords"].detach().float()
    plddt = output["plddt"].detach().float()
    if coords.dim() == 3:
        coords = coords[0]
    coords_b = coords.unsqueeze(0)

    if plddt.dim() == 3:
        plddt = plddt[0]
    if plddt.dim() == 2:
        plddt = plddt[0]

    fd, path = tempfile.mkstemp(suffix="_boltz_refold.pdb")
    os.close(fd)
    try:
        save_pdb(structure, coords_b, plddt, path)
        return io.load_structure(path)
    finally:
        Path(path).unlink(missing_ok=True)


def get_cif(cif_code=""):
    """
    Returns the local filename (relative path) to the CIF for the provided code or file.
    Checks local directory and fetches from RCSB or AlphaFold if needed.
    """
    if cif_code is None or cif_code == "":
        print("Error: No cif code specified and uploads not supported in CLI mode.")
        sys.exit(1)
    elif os.path.isfile(cif_code):
        return os.path.abspath(cif_code)
    elif len(cif_code) == 4:
        # PDB ID
        local_cif = f"{cif_code}.cif"
        if not os.path.isfile(local_cif):
            os.system(f"wget -qnc https://files.rcsb.org/download/{cif_code}.cif")
        return os.path.abspath(local_cif)
    else:
        # AlphaFold ID
        local_cif = f"AF-{cif_code}-F1-model_v3.cif"
        if not os.path.isfile(local_cif):
            os.system(
                f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{cif_code}-F1-model_v3.cif"
            )
        return os.path.abspath(local_cif)


def get_CA(x):
    """
    DEPRECATED: Use get_CA_and_sequence from metrics instead.
    Returns CA coordinates from a PDB file.
    """
    xyz = []
    with open(x) as file:
        for line in file:
            line = line.rstrip()
            if line[:4] == "ATOM":
                atom = line[12 : 12 + 4].strip()
                if atom == "CA":
                    x_coord = float(line[30 : 30 + 8])
                    y_coord = float(line[38 : 38 + 8])
                    z_coord = float(line[46 : 46 + 8])
                    xyz.append([x_coord, y_coord, z_coord])
    return np.array(xyz)


def binder_binds_contacts(
    pdb_path, binder_chain, target_chain, contact_residues, cutoff=10.0
):
    """
    Returns True if at least 2 contact residues on target_chain
    are contacted by any CA atom of binder_chain within cutoff angstroms.
    """
    if isinstance(contact_residues, str):
        if contact_residues.strip() == "":
            return True
        contact_residues = [int(x) for x in contact_residues.split(",") if x.strip()]

    structure = parsePDB(pdb_path)
    if structure is None:
        return False

    def get_chain_ca_atoms_and_resnums(chain_id):
        ca_atoms = []
        ca_resnums = []
        for atom in structure.iterAtoms():
            # Check chain ID and atom name
            if atom.getChid() == chain_id and atom.getName() == "CA":
                ca_atoms.append(atom)
                ca_resnums.append(atom.getResnum())
        return ca_atoms, ca_resnums

    binder_ca_atoms, _ = get_chain_ca_atoms_and_resnums(binder_chain)
    target_ca_atoms, target_resnums = get_chain_ca_atoms_and_resnums(target_chain)

    if len(binder_ca_atoms) == 0 or len(target_ca_atoms) == 0:
        return False

    binder_coords = np.array([atom.getCoords() for atom in binder_ca_atoms])
    target_coords = np.array([atom.getCoords() for atom in target_ca_atoms])


    contact_indices = [
        i for i, resnum in enumerate(target_resnums) if resnum in contact_residues
    ]
    if not contact_indices:
        return False

    filtered_target_coords = target_coords[contact_indices]
    
    # For each contact residue, check if any binder CA is within cutoff.
    contacted = 0
    for c_coord in filtered_target_coords:
        distances = np.sqrt(np.sum((binder_coords - c_coord) ** 2, axis=-1))
        if np.any(distances < cutoff):
            contacted += 1

    # Require at least 2 contacted residues to pass the filter
    return contacted >= 2


def sample_seq(length: int, exclude_P: bool = True, frac_X: float = 0.0) -> str:
    """Samples a random sequence of the given length, optionally excluding Proline (P) and including 'X' residues."""
    aas = "ACDEFGHIKLMNQRSTVWY" + ("" if exclude_P else "P")
    num_x = round(length * frac_X)
    pool = aas if aas else "X"
    seq_list = ["X"] * num_x + random.choices(pool, k=length - num_x)
    random.shuffle(seq_list)
    return "".join(seq_list)


def extract_sequence_from_structure(pdb_path, chain_id):
    """
    Extract 1-letter sequence from PDB file for a given chain.
    (DEPRECATED: Use get_CA_and_sequence and take the sequence part instead)
    """
    try:
        _, sequence = get_CA_and_sequence(pdb_path, chain_id)
        return sequence
    except Exception as e:
        raise ValueError(f"Could not extract sequence from chain {chain_id} in {pdb_path}: {e}") from e


def shallow_copy_tensor_dict(d):
    """Performs a shallow copy of a nested dictionary, cloning only torch.Tensors."""
    if isinstance(d, dict):
        return {k: shallow_copy_tensor_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [shallow_copy_tensor_dict(x) for x in d]
    if isinstance(d, torch.Tensor):
        return d.detach().clone()
    return d


def smart_split(s):
    if "," in s:
        return [x.strip() for x in s.split(",")]
    if ":" in s:
        return [x.strip() for x in s.split(":")]
    return [x.strip() for x in s.split()] if s else []


def get_boltz_model(
    checkpoint: Optional[str] = None,
    predict_args: Optional[dict] = None,
    device: Optional[str] = None,
    model_version: str = "boltz2",
    grad_enabled=True,
    no_potentials=True,
) -> Boltz1 | Boltz2:
    """Loads and configures the Boltz model based on arguments."""
    if predict_args is None:
        predict_args = default_boltz_predict_args()

    torch.set_grad_enabled(grad_enabled)
    torch.set_float32_matmul_precision("highest")
    
    # Setup diffusion parameters
    diffusion_params = Boltz2DiffusionParams()
    diffusion_params.step_scale = 1.638

    # Setup steering parameters
    steering_args = BoltzSteeringParams()
    if no_potentials:
        print("no potentials")
        steering_args.fk_steering = False
        steering_args.guidance_update = False
    else:
        print("use potentials")

    # Setup pairformer arguments
    pairformer_args = (
        PairformerArgsV2() if model_version == "boltz2" else PairformerArgs()
    )
    pairformer_args.v2 = model_version == "boltz2"
    pairformer_args.activation_checkpointing = True

    # Setup MSA module arguments
    msa_args = MSAModuleArgs(
        subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True
    )
    msa_args.activation_checkpointing = True

    # Load model from checkpoint
    model_class = Boltz2 if model_version == "boltz2" else Boltz1
    
    if model_version == "boltz2":
        model_module = model_class.load_from_checkpoint(
            checkpoint,
            strict=False,
            predict_args=predict_args,
            map_location=device,
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            structure_prediction_training=True,
            no_msa=False,
            no_atom_encoder=False,
            use_templates=True,
            use_templates_v2=True,
            use_trifast=False,
            max_parallel_samples=1,
            steering_args=asdict(steering_args),
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            weights_only=False,
        )
    elif model_version == "boltz1":
        model_module = Boltz1.load_from_checkpoint(
            checkpoint,
            strict=False,
            predict_args=predict_args,
            map_location=device,
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            structure_prediction_training=True,
            no_msa=False,
            no_atom_encoder=False,
        )
    else:
        raise ValueError(f"Unknown Boltz model version: {model_version}")

    return model_module

def get_batch(
    target,
    ccd_path,
    ccd_lib,
    max_seqs=0,
    pocket_conditioning=False,
    keep_record=False,
    boltz_model_version=None,
):
    max_seqs = 4096
    structure = target.structure

    coords = np.array([(atom["coords"],) for atom in structure.atoms], dtype=Coords)
    ensemble = np.array([(0, len(coords))], dtype=Ensemble)

    if boltz_model_version == "boltz2":
        structure = StructureV2(
            atoms=structure.atoms,
            bonds=structure.bonds,
            residues=structure.residues,
            chains=structure.chains,
            interfaces=structure.interfaces,
            mask=structure.mask,
            coords=coords,
            ensemble=ensemble,
        )

    elif boltz_model_version == "boltz1":
        structure = Structure(
            atoms=structure.atoms,
            bonds=structure.bonds,
            residues=structure.residues,
            chains=structure.chains,
            interfaces=structure.interfaces,
            mask=structure.mask,
            connections=structure.connections,
        )

    msas = {}
    for chain in target.record.chains:
        msa_id = chain.msa_id
        if msa_id != -1:
            msa = np.load(msa_id)
            msas[chain.chain_id] = MSA(**msa)

    input = Input(
        structure,
        msas,
        record=target.record,
        residue_constraints=target.residue_constraints,
        templates=target.templates,
        extra_mols=target.extra_mols,
    )

    if boltz_model_version == "boltz2":
        tokenizer = Boltz2Tokenizer()
        featurizer = Boltz2Featurizer()
    elif boltz_model_version == "boltz1":
        tokenizer = BoltzTokenizer()
        featurizer = BoltzFeaturizer()

    tokenized = tokenizer.tokenize(input)

    # seed = 42
    # random = np.random.default_rng(seed)
    random = np.random.default_rng()
    if boltz_model_version == "boltz2":
        molecules = {}
        molecules.update(ccd_lib)
        molecules.update(input.extra_mols)
        mol_names = set(tokenized.tokens["res_name"].tolist())
        mol_names = mol_names - set(molecules.keys())
        molecules.update(load_molecules(ccd_path, mol_names))
    options = target.record.inference_options
    if pocket_conditioning:
        pocket_constraints, contact_constraints = (
            options.pocket_constraints,
            options.contact_constraints,
        )
        if boltz_model_version == "boltz2":
            batch = featurizer.process(
                tokenized,
                random=random,
                molecules=molecules,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=max_seqs,
                pad_to_max_seqs=False,
                compute_symmetries=False,
                single_sequence_prop=0.0,
                compute_frames=True,
                inference_pocket_constraints=pocket_constraints,
                inference_contact_constraints=contact_constraints,
                compute_constraint_features=True,
                compute_affinity=False,
            )
        elif boltz_model_version == "boltz1":
            binders, pocket = options.binders, options.pocket
            batch = featurizer.process(
                tokenized,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=max_seqs,
                pad_to_max_seqs=False,
                symmetries={},
                compute_symmetries=False,
                inference_binder=binders,
                inference_pocket=pocket,
            )

    else:
        pocket_constraints = None

        if boltz_model_version == "boltz2":
            batch = featurizer.process(
                tokenized,
                random=random,
                molecules=molecules,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=max_seqs,
                pad_to_max_seqs=False,
                compute_symmetries=False,
                single_sequence_prop=0.0,
                compute_frames=True,
                inference_pocket_constraints=pocket_constraints,
                compute_constraint_features=True,
                compute_affinity=False,
            )
        else:
            batch = featurizer.process(
                tokenized,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=max_seqs,
                pad_to_max_seqs=False,
                symmetries={},
                compute_symmetries=False,
                inference_binder=None,
                inference_pocket=None,
            )

    if keep_record:
        batch["record"] = target.record

    return batch, structure

def process_msa(chain_id: str, sequence: str, msa_dir: Path) -> Path:
    """Run ColabFold/MMseqs2 for ``sequence`` and return the path to Boltz-ready ``.npz`` MSA.

    Use the returned path in Boltz YAML / :func:`build_protein_boltz_yaml` ``msa_paths`` for that
    ``chain_id`` (Boltz loads it via :func:`numpy.load` during featurization).
    """
    msa_chain_dir = msa_dir / f"{chain_id}"
    env_dir = msa_chain_dir.with_name(f"{msa_chain_dir.name}_env")
    env_dir.mkdir(exist_ok=True)

    # Run MSA
    unpaired_msa = run_mmseqs2(
        [sequence],
        str(msa_chain_dir),
        use_env=True,
        use_pairing=False,
        host_url="https://api.colabfold.com",
        pairing_strategy="greedy",
    )

    # Save MSA results (.a3m)
    msa_a3m_path = env_dir / "msa.a3m"
    msa_a3m_path.write_text(unpaired_msa[0])

    # Process MSA (.a3m) into Boltz .npz format
    msa_npz_path = env_dir / "msa.npz"
    if not msa_npz_path.exists():
        msa = parse_a3m(
            msa_a3m_path,
            taxonomy=None,
            max_seqs=4096,
        )
        msa.dump(msa_npz_path)

    return msa_npz_path


def aggressive_memory_cleanup():
    """Performs aggressive CUDA and Python memory cleanup."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.synchronize()

    for _ in range(3):
        gc.collect()

    # Reset torch dynamo cache if available
    if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'reset'):
        torch._dynamo.reset()
    
    # Clear cublas workspaces if function exists
    if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
        torch._C._cuda_clearCublasWorkspaces()


def clean_memory():
    """Wrapper for general garbage collection and aggressive cleanup."""
    gc.collect()
    torch.cuda.empty_cache()
    aggressive_memory_cleanup()


def run_prediction(
    data,
    binder_chain: str | None = None,
    seq=None,
    logmd=False,
    name: str | None = None,
    ccd_lib=None,
    ccd_path=None,
    boltz_model=None,
    randomly_kill_helix_feature=False,
    negative_helix_constant=0.1,
    device="cpu",
    boltz_model_version="boltz2",
    pocket_conditioning=False,
):
    """Parses data, generates batch, and runs a single Boltz prediction step."""
    if seq is not None and binder_chain is not None:
        found = False
        for entry in data.get("sequences", []):
            prot = entry.get("protein")
            if not prot:
                continue
            ids = prot["id"]
            if isinstance(ids, str):
                ids = [ids]
            if binder_chain in ids:
                prot["sequence"] = seq
                found = True
                break
        if not found:
            raise KeyError(
                f"Chain {binder_chain!r} not found under data['sequences'] for sequence update."
            )

    target = parse_boltz_schema(
        name or "design",
        data,
        ccd_lib,
        ccd_path,
        boltz_2=boltz_model_version == "boltz2",
    )

    batch, structure = get_batch(
        target,
        ccd_path,
        ccd_lib,
        boltz_model_version=boltz_model_version,
        pocket_conditioning=pocket_conditioning,
    )

    batch = batch_to_device(batch, device)

    output = boltz_model.predict_step(batch, batch_idx=0, dataloader_idx=0)
    return output, structure


def save_pdb(structure, coords, plddts, filename):
    """Saves the predicted structure coordinates to a PDB file."""
    structure.atoms["coords"] = (
        coords[0].detach().cpu().numpy()[: structure.atoms["coords"].shape[0]]
    )
    
    with open(filename, "w") as f:
        f.write(to_pdb(structure, plddts, boltz2=True))


def design_sequence(
    designer,
    model_type,
    pdb_file,
    chains_to_design="A",
    omit_AA="C",
    bias_AA="",
    temperature=0.02,
    return_logits=False,
    fixed_residues="",
    temperature_per_residue=None,
    extra_args={},
):
    """Runs the LigandMPNN (or SolubleMPNN) sequence design wrapper."""
    seq, logits = designer.run(
        model_type=model_type,
        pdb_path=pdb_file,
        seed=111, # Fixed seed for reproducibility per design tool
        chains_to_design=chains_to_design,
        bias_AA=bias_AA,
        omit_AA=omit_AA,
        fixed_positions=fixed_residues,
        temperature=temperature,
        temperature_per_residue=temperature_per_residue,
        return_logits=return_logits,
        extra_args=extra_args,
    )
    if return_logits:
        return seq, logits

    return seq, logits


def plot_from_pdb(
    pdb_file: str,
    width: int = 400,
    height: int = 400,
    style: str = "cartoon",
    color_by: str = "plddt",
):
    """
    Visualize a structure directly from a PDB file using py3Dmol.
    """
    pdb_text = Path(pdb_file).read_text()

    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_text, "pdb")

    color_by = (color_by or "none").lower()

    if color_by == "plddt":
        view.setStyle(
            {"and": [{"not": {"resn": ["UNL", "LIG"]}}, {"not": {"hetflag": True}}]},
            {
                style: {
                    "colorscheme": {
                        "prop": "b",
                        "gradient": "roygb",
                        "min": 50,
                        "max": 90,
                    }
                }
            },
        )
    # Color by chain
    elif color_by == "chain":
        view.setStyle(
            {"and": [{"not": {"resn": ["UNL", "LIG"]}}, {"not": {"hetflag": True}}]},
            {style: {"colorscheme": "chain"}},
        )
    # Default coloring
    else:
        view.setStyle(
            {"and": [{"not": {"resn": ["UNL", "LIG"]}}, {"not": {"hetflag": True}}]},
            {style: {}},
        )

    # Show ligands/heteroatoms as sticks
    view.setStyle(
        {"or": [{"resn": ["UNL", "LIG"]}, {"hetflag": True}]},
        {"stick": {"radius": 0.2}},
    )

    view.zoomTo()
    return view.show()


def plot_run_metrics(
    run_save_dir: str, name: str, run_id: int, num_cycles: int, run_metrics: dict
):
    """Plots per-run metrics (iPTM, pLDDT, Alanine Count) over design cycles."""
    fig, axs = plt.subplots(1, 4, figsize=(16, 4)) # Increased figure width
    colors = ["#9B59B6", "#E94560", "#FF7F11", "#2ECC71"]
    
    # Helper to retrieve data and format
    def get_metric_data(key_suffix, label, ymin, ymax, fmt):
        values = [run_metrics.get(f"cycle_{i}_{key_suffix}", float("nan")) for i in range(num_cycles + 1)]
        return (label, values, ymin, ymax, fmt)

    metrics_list = [
        get_metric_data("iptm", "iPTM", 0, 1, "{:.3f}"),
        get_metric_data("plddt", "pLDDT", 0, 1, "{:.1f}"), # Corrected pLDDT max to 100
        get_metric_data("iplddt", "iPLDDT", 0, 1, "{:.1f}"), # Corrected iPLDDT max to 100
        get_metric_data(
            "alanine", 
            "Alanine Count", 
            0, 
            max([run_metrics.get(f"cycle_{i}_alanine", 0) for i in range(num_cycles + 1)]) + 2, 
            "{}",
        ),
    ]
    
    design_cycles = list(range(num_cycles + 1))
    
    for ax, (label, values, ymin, ymax, fmt), color in zip(axs, metrics_list, colors):
        valid_indices = [i for i, y in enumerate(values) if not pd.isnull(y)]
        valid_cycles = [design_cycles[i] for i in valid_indices]
        valid_values = [values[i] for i in valid_indices]

        ax.plot(
            valid_cycles,
            valid_values,
            "o-",
            color=color,
            linewidth=2,
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=2,
        )
        ax.set(
            xlabel="Design Iteration",
            ylabel=label,
            title=f"{label} (Run {run_id})",
            xticks=design_cycles,
            ylim=(ymin, ymax),
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        
        # Annotate points
        for x, y in zip(design_cycles, values):
            if not pd.isnull(y):
                 ax.annotate(
                    fmt.format(y),
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=9,
                )
                
    plt.tight_layout()
    plot_filename = f"{name}_run_{run_id}_design_cycle_results.png"
    plt.savefig(f"{run_save_dir}/{plot_filename}", dpi=300)
    plt.show(block=False)
