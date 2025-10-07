# RICA Project Structure

This document describes the structure of the RICA (Re-Ranking with Intra-Modal and Cross-Modal Alignment) project.

## Core Files

- `main.py` - Main entry point for training
- `train.py` - Training script
- `test.py` - Testing script
- `run_rica.sh` - Training script with default parameters
- `run_test.sh` - Testing script

## Directory Structure

```
RICA-release/
├── model/                    # Model definitions
│   ├── build.py             # Main RICA model class
│   ├── clip_model.py        # CLIP-based backbone
│   ├── losses.py            # Loss functions
│   └── objectives.py        # Objective functions
├── datasets/                 # Dataset handling
│   ├── build.py             # Dataset builder
│   ├── bases.py             # Base dataset class
│   ├── cuhkpedes.py         # CUHK-PEDES dataset
│   ├── icfgpedes.py         # ICFG-PEDES dataset
│   ├── rstpreid.py          # RSTPReid dataset
│   ├── iiitd_20k.py         # IIITD-20K dataset
│   ├── preprocessing.py     # Data preprocessing
│   ├── sampler.py           # Data sampling
│   └── sampler_ddp.py       # Distributed data sampling
├── processor/                # Training processors
│   └── processor.py         # Training and inference logic
├── solver/                   # Optimizers and schedulers
│   ├── build.py             # Optimizer builder
│   └── lr_scheduler.py      # Learning rate scheduler
├── utils/                    # Utility functions
│   ├── checkpoint.py        # Model checkpointing
│   ├── comm.py              # Communication utilities
│   ├── iotools.py           # I/O utilities
│   ├── logger.py            # Logging utilities
│   ├── meter.py             # Performance meters
│   ├── metrics.py           # Evaluation metrics
│   ├── options.py           # Command line options
│   └── simple_tokenizer.py  # Text tokenizer
└── data/                     # Data directory
    └── bpe_simple_vocab_16e6.txt.gz  # BPE vocabulary
```

## Key Components

### Model Architecture
- **RICA Class**: Main model class implementing the re-ranking approach
- **CLIP Backbone**: Vision and text encoders based on CLIP
- **Cross-Modal Transformer**: For cross-modal feature alignment
- **Alignment Modules**: Intra-modal and cross-modal alignment losses

### Loss Functions
- **Identity Loss**: Person identity classification
- **Global-Local Alignment Loss**: Intra-modal alignment
- **Cross-Modal Alignment Loss**: Cross-modal alignment

### Datasets
- **CUHK-PEDES**: Primary dataset for evaluation
- **ICFG-PEDES**: Additional dataset for validation
- **RSTPReid**: Real-world surveillance dataset

## Usage

1. **Training**: `bash run_rica.sh [dataset_name] [gpu_id]`
2. **Testing**: `bash run_test.sh [config_file] [output_dir] [gpu_id]`
3. **Custom Training**: Modify parameters in `run_rica.sh` or use `train.py` directly

## Configuration

Key parameters can be modified in `utils/options.py`:
- `--delta`: Margin for hardest pairs (default: 4.0)
- `--sigma`: Margin for alignment (default: 0.08)
- `--loss_names`: Loss functions to use (default: "id+gl+mlm")
- `--batch_size`: Training batch size (default: 64)
- `--lr`: Learning rate (default: 1e-5)
