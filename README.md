# ELIQSIR-QDL

**Equivariant Layers for Interactions & Quantum-State Inference Research**

This repository is an experimental framework for predicting protein-ligand binding affinity (`pKd`) using a hybrid architecture that combines:
- classical graph and sequence encoders,
- a flexible multi-slot encoder pipeline,
- an optional quantum perturbation branch powered by PennyLane,
- an adaptive mixer that learns how to combine classical and quantum outputs.

## 1. Project overview

`ELIQSIR-QDL` is designed for research, not production. It explores how structure-aware representations of proteins, ligands, and binding pockets can be fused with quantum-inspired correction terms.

The core idea is:
- build a shared embedding from protein, ligand, and pocket information,
- compute a classical affinity estimate,
- optionally compute a quantum-derived affinity shift,
- learn a final combination of those two contributions.

## 2. Molecular physics and graph meaning

### 2.1 Why molecules are graphs

Molecules are naturally represented as graphs because atoms and interactions form discrete objects and relationships:
- nodes represent atoms, residues, or local molecular features,
- edges represent chemical bonds, spatial proximity, or contact relationships,
- node features include atomic coordinates, element type, hybridization degree, aromaticity, and other local descriptors.

In this project:
- proteins are represented as graphs of Cα atoms or residue centers based on 3D coordinates,
- ligands are represented as chemical graphs built from RDKit bonds and atom properties,
- pockets are represented as local graphs around the binding site, capturing the interaction environment.

### 2.2 Physics of binding affinity

Binding affinity is fundamentally a physical quantity that reflects how strongly two molecules interact.
- higher `pKd` means stronger binding,
- affinity depends on geometry, hydrogen bonding, hydrophobic contacts, and steric complementarity,
- the goal is to infer this energy-like signal from structural and topological features.

This repository separates two conceptual contributions:
1. a classical energy estimate computed from the learned graph/sequence embedding,
2. an optional quantum-inspired perturbation component that adjusts the final prediction.

### 2.3 Why modular encoders matter

The project is not limited to one fixed encoder type. The encoder pipeline is modular:
- the current default uses a `trio_encoder` for protein, ligand, and pocket,
- the same architecture can later support a `duo_encoder` that outputs two fused representations, for example `protein` and `ligand+pocket`,
- each slot can be a CNN or a GNN, depending on the chosen config string.

This makes the system flexible for experiments with different inductive biases and interaction patterns.

## 3. Model architecture

The main model class is `UniversalHybridSlotModel` in `model.py`.

It is built from configurable slots:
- `graph_encoder` — slot 1, encodes the input tuple (protein, ligand, pocket) into a shared representation,
- `classic_pooler` — slot 2A, optional classical pooling or selection of graph nodes,
- `quantum_pooler` — slot 2B, optional pooling to select nodes or features for the quantum branch,
- `global_readout` — slot 3, aggregates node embeddings into a global vector,
- `quantum_encoder` — slot 4, maps the classical adapter output into a quantum feature vector.

### 3.1 Branching logic

The classical branch always exists and produces `base_affinity`.
When a quantum encoder is provided, the model also computes a `quantum_affinity_shift` and learns how to mix both values.
If no quantum branch is present, the model returns the classical affinity directly.

### 3.2 Block diagram

```text
                            [protein, ligand, pocket]
                                    │
                                    ▼
                        ┌──────────────────────────┐
                        │      graph_encoder       │
                        │ (trio / duo / EGNN, ...) │
                        └──────────────────────────┘
                                    │
                                    ▼
                            ┌──────────────────┐
                            │ classical branch │
                     ┌──────│  global_readout  │──────┐
                     |      │  classic_decider │      |
                     |      └──────────────────┘      |
                     │                                |
                     │                                |
                     │                                |
                     |                                │
                     ▼                                ▼
            ┌──────────────────┐        ┌───────────────────────────┐
            │  quantum_pooler  │        | classic_pooler (optional) |
            ┌──────────────────┐        ┌───────────────────────────┐
            │ quantum branch   │        │  Multi-Layer Perceptron   │
            │  adapter Linear  │        └───────────────────────────┘
            │  quantum_encoder │                      │
            │  quantum_decider │                      │
            └──────────────────┘                      │
                     │                                │
                     │                                │
         quantum_shift (optional)                base_affinity
                     │                                │
                     ▼                                ▼
                  ┌──────────────────────────────────────┐
                  │             learnable mixer          │
                  │            concat([base, q])         │
                  │              Linear(2->1)            │
                  └──────────────────────────────────────┘
                                     │
                                     ▼
                         final affinity prediction
```

This diagram shows the branching behavior clearly:
- the classical path always produces an affinity estimate,
- the optional quantum path provides a corrective shift,
- the final mixer learns how to weigh both signals.

## 4. Main components

| File | Description |
|---|---|
| `model.py` | The universal hybrid model combining classical and quantum branches. |
| `extractor.py` | PDBBind orchestration: archive extraction, complex selection, parsing, and dataset export. |
| `tokenizer.py` | Dataset loader that accepts CSV/Pickle/Parquet and builds PyTorch/PyG inputs. |
| `encoders/gnn_encoder.py` | A flexible GNN block supporting GCN, GAT, GATv2, and pooling. |
| `encoders/cnn_encoder.py` | A CNN block for sequence-like inputs (SMILES, amino acids). |
| `encoders/classic_trio_encoder.py` | The encoder factory for protein/ligand/pocket slots and encoder configuration strings. |
| `encoders/original_quantum_encoder.py` | PennyLane quantum re-uploading layer. |
| `parsers/gnn_parser.py` | Graph parsers for PDB proteins and ligand SDF structures. |
| `requirements.txt` | Python dependency list. |

## 5. Graph pipeline flow

1. `PDBBindOrchestrator` in `extractor.py`:
   - extracts the PDBBind archive,
   - reads index files `INDEX_*.2016`,
   - selects complexes with `res <= 2.5` and valid `pKd`,
   - parses protein, ligand, and pocket structures,
   - saves the dataset in `parquet`, `pickle`, or `csv`.

2. `GNNParser` in `parsers/gnn_parser.py` converts:
   - protein structures into Cα coordinate graphs,
   - ligands into chemical graphs with atom features,
   - pockets into local graphs around the binding site.

3. `UniversalPDBBindDataset` in `tokenizer.py`:
   - loads tabular dataset formats automatically,
   - turns each row into PyTorch data objects,
   - returns `(protein, ligand, pocket, pkd)` examples.

## 6. Encoder flexibility

The encoder design is intentionally modular:
- `TrioPipelineFactory` builds matching parser and encoder tuples using symbols like `C` for CNN and `G` for GNN,
- current configs such as `CCC`, `CGC`, or `GCG` show that each slot can be chosen independently,
- future extensions may implement `duo_encoder`, where protein and fused ligand+pocket are encoded by separate branches,
- this allows experimentation with different inductive biases and architecture families.

## 7. Quantum branch details

The quantum block in `encoders/original_quantum_encoder.py` uses a `QuantumReUploadingLayer`.

Key behavior:
- inputs are projected to the quantum dimension with a linear adapter,
- the layer applies `AngleEmbedding` and several `StronglyEntanglingLayers`,
- output is a vector of expectation values `⟨Z⟩` for each qubit,
- those values are reduced by `quantum_decider` to a scalar shift.

This branch is optional, so the model remains usable without a quantum encoder.

## 8. Typical usage pattern

1. Prepare the data:
   - extract and filter PDBBind,
   - choose a subset such as `refined`, `core`, or `general`,
   - build a dataset via `PDBBindOrchestrator.build_dataset()`.

2. Build the encoder pipeline:
   - choose a configuration string, e.g. `"CGC"`, `"GCG"`, or `"CCC"`,
   - use `TrioPipelineFactory.build(config_str, config_dict)` to get parsers and encoders.

3. Instantiate the model:
   - `graph_encoder = SplitTrioEncoder(...)`,
   - `quantum_encoder = QuantumReUploadingLayer(...)`,
   - `model = UniversalHybridSlotModel(...)`.

4. Train and evaluate:
   - load data through PyTorch `DataLoader`,
   - treat `pkd` as a regression target,
   - optimize the final mixer and the branch parameters.

## 9. Deployment and environment setup (optional)

> This section is for fast project setup on a new machine.

### Create Python environment

```bash
cd /home/yaroslav/FCUL/ELIQSIR-QDL
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you use GPU, install PyTorch from the official site: `https://pytorch.org`.

### Environment structure

- `.venv/` — local Python virtual environment,
- `requirements.txt` — dependency list,
- `data/` and `datasets/` — raw and parsed dataset files,
- `runs/` — saved experiment outputs.

### Quick check

```bash
source .venv/bin/activate
python -c "from extractor import PDBBindOrchestrator; print('ok')"
```

### Manual dependency install

```bash
pip install rdkit biopython pandas tqdm ipywidgets matplotlib torch torch_geometric pennylane
```

> Note: `torch_geometric` is often installed separately and depends on the installed PyTorch and CUDA version.

## 10. Important notes

- The repo is research-oriented, not a production-ready application.
- There are experimental design choices: multiple encoder configurations, hybrid classical/quantum logic, and alternative pooling strategies.
- The PDBBind archive `pdbbind_v2016.tar.gz` must be available in the project root or referenced by the data loader.

## 11. Useful module summary

- `extractor.py` — dataset orchestration and PDBBind parsing.
- `tokenizer.py` — flexible dataset loader for graph and sequence inputs.
- `encoders/gnn_encoder.py` — graph neural network block implementation.
- `encoders/cnn_encoder.py` — CNN-based sequence encoder.
- `encoders/classic_trio_encoder.py` — slot-based encoder assembly for protein/ligand/pocket.
- `encoders/original_quantum_encoder.py` — quantum re-uploading layer.

## 12. Suggested next steps

1. Add a runnable training script such as `train.py`.
2. Add a minimal example notebook or script showing `TrioPipelineFactory` and `UniversalPDBBindDataset` usage.
3. Add a `duo_encoder` or other multi-slot alternatives as a documented extension.
4. Add Docker or Makefile support for reproducible setup.
5. Add a temperature as a parameter for the final mixer.

---
