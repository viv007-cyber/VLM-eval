## 📂 Repository Layout & Architectural Mapping

```text
├── configs/                         # Model and dataset orchestration configuration profiles
│   ├── dataset/                     # Task-specific pipeline dataset loaders
│   └── model/                       # Quantization, precision, and GPU device routing states
│
├── data/                            # Raw Medical Imagery Matrix (Ground Truth Grounding Targets)
│   ├── MRI/
│   │   └── Knee/                    # Knee MRI scans partitioned by anatomical cuts and pathologies
│   │       ├── Axial/               # Cross-sectional slices (e.g., 0001.png, 0009.png)
│   │       ├── Coronal/             # Frontal plane view cuts (e.g., 0012.png)
│   │       ├── Sagittal/            # Side profile plane views (e.g., 0005.png)
│   │       └── meniscal_tears/      # Focused structural pathology scans (e.g., 0034.png)
│   └── Xray/
│       ├── Hip/                 # Hip X-rays (Posteroanterior view, MSD disorders)
│       ├── Knee/                # Knee X-rays (Anteroposterior view, Knee Osteoarthritis)
│       └── Shoulder/            # Shoulder X-rays (Anteroposterior view, Cortical Fractures)
│
├── outputs/                         # Complete Multi-VLM Target Inference Log Arrays (.csv)
│   ├── InstructBlip/                # Domain-wise predictions generated via InstructBLIP
│   ├── LLava/                       # Evaluation response vectors mapped from LLaVA-1.5
│   ├── MedGemma/                    # Baseline evaluations for MedGemma-4B
│   ├── Medgemma27B/                 # Specialized large-scale medical evaluations (MRI vs. X-ray splits)
│   ├── Paligemma/                   # Downstream task output tracking for PaliGemma
│   └── Qwen/                        # Sequence metrics computed using Qwen2.5-VL benchmarks
│
├── prompts/
│   └── eval1/                       # Modular evaluation prompt structures saved as JSON states
│       ├── anatomy.json             # Targeted structures queries ("Identify structural region...")
│       ├── modality.json            # Imaging method isolation ("Is this CT, MRI, or CR...")
│       ├── pathology.json           # Diagnostic classification targets ("Locate lesion...")
│       └── view.json                # Scan perspective mappings ("Determine projection orientation...")
│
├── src/vlm_eval/                    # Core Execution Engine Package
│   ├── diagnostics/                 # Interpretability, rollout hooks, and text mutation algorithms
│   │   └── generate_descriptions.py # Gemini engine running automated counterfactual synthesis loops
│   ├── models/                      # Multi-VLM Model Adapters (The Unified Structural Interface)
│   │   ├── base.py                  # Abstract base class declaring inference and attention patterns
│   │   ├── instructblip.py          # Adapter wrapping FLAN-T5/Vicuna Q-Former lookups
│   │   ├── llava_model.py           # Adapter managing LLaVA linear projection tracking
│   │   ├── medgemma.py              # Adapter routing MedGemma-4B & 27B dual-tower frameworks
│   │   ├── paligemma.py             # Adapter handling SigLIP-to-Gemma cross-token configurations
│   │   └── qwen_vl.py               # Adapter running memory-capped Qwen dynamic resolution grids
│   ├── tasks/                       # Downstream validation metric calculation routines
│   └── utils/                       # File system I/O utilities and sequence token helpers
│
├── tests/                           # Verification unit arrays containing target test image assertions
├── pyproject.toml                   # Strict build-system workspace matrix and dependencies
└── README.md                        # Documentation hub
