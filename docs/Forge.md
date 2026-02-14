# Forge System Setup & Architecture

**Machine Name**: Forge  
**Purpose**: Dedicated headless ML workstation for neural pruning research  
**Status**: Production-ready infrastructure for Project Bonsai and related ML research  

---

## 🖥️ Hardware Specifications

### Core Components
- **CPU**: AMD Ryzen 5 7600X (6 cores / 12 threads, up to ~5.4 GHz)
- **RAM**: 64 GB DDR5 (4 x 16 GB, running at 6000 MHz via EXPO)
- **GPU**: NVIDIA GeForce RTX 5070 Ti (16 GB VRAM, Ada Lovelace architecture)
- **Motherboard**: ASUS Prime B650M-A AX II (supports DDR5 and PCIe Gen4)
- **PSU**: Lian-Li EGO750G
- **Cooling**: ThermalRight Peerless Assassin 120 SE CPU Air Cooler

### Performance Characteristics
- **CUDA Capability**: Full support for modern ML frameworks
- **Memory Bandwidth**: High-speed DDR5 for large dataset processing
- **Storage**: Dual NVMe Gen4 SSDs for optimal I/O performance
- **Network**: Integrated for remote development workflows

---

## 💾 Storage Architecture

Forge uses an optimized dual-drive setup designed specifically for ML workloads:

### SSD 1 (2TB) - System & Projects
```
/               (~1 TB)     - Ubuntu 24.04 LTS install, conda environments
/mnt/projects/  (~800 GB)   - Source code, notebooks, configurations
```

### SSD 2 (2TB) - Data & Artifacts  
```
/mnt/data/      (1.5 TB)    - Raw datasets, databases, shared storage
/mnt/artifacts/ (~500 GB)   - Trained model artifacts, checkpoints
```

### Storage Philosophy
- **Projects drive**: Source code and configurations (version controlled)
- **Data drive**: Raw datasets and scratch files (high I/O)
- **Artifacts drive**: Model outputs and results (persistent storage)

---

## 📂 Directory Organization

### Common Datasets Structure
```
/mnt/data/common_datasets/
├── wine/           # Wine quality dataset (UCI ML repository)
├── cifar10/        # CIFAR-10 computer vision benchmark
├── mnist/          # MNIST handwritten digits
└── [other]/        # Additional canonical datasets
```

### Project-Specific Structure (Bonsai Example)
```
/mnt/data/bonsai/
├── quick/
│   ├── datasets/     # Symlinks to common_datasets
│   ├── models/       # Trained model files (.pth, .pkl)
│   ├── nim_data/     # FANIM/BANIM sensitivity data (.h5)
│   └── scratch/      # Temporary processing files
└── ibonsai/
    ├── models/       # iBonsai-specific trained models
    ├── sensitivity/  # Advanced sensitivity analysis
    └── datasets/     # Project-specific dataset variations
```

### Configuration Management
```
/mnt/projects/project_bonsai/
├── quick/
│   ├── configs/      # Experiment configurations
│   ├── notebooks/    # Jupyter analysis notebooks
│   └── shared/       # Common utilities and forge_config.py
└── archive/          # Historical research code
```

---

## 🕹️ Software Stack

### Operating System
- **OS**: Ubuntu 24.04.2 LTS (minimal, headless installation)
- **Kernel**: Linux with optimizations for ML workloads
- **Boot**: UEFI with secure boot disabled for research flexibility

### GPU & CUDA Environment
- **Driver**: NVIDIA driver 570 series (open kernel driver for Ada GPUs)
- **CUDA**: Latest CUDA toolkit compatible with RTX 5070 Ti
- **Libraries**: cuDNN, cuBLAS optimized for deep learning
- **Memory**: 16GB VRAM supports large model training and inference

### Python & ML Environment
- **Package Manager**: Conda for environment isolation
- **Primary Environment**: `ml` environment for Python-based ML work
- **Key Libraries**: PyTorch, NumPy, SciPy, scikit-learn, h5py
- **Frameworks**: Support for PyTorch, TensorFlow, JAX

### Database Infrastructure
- **Database**: PostgreSQL 16.9
- **Data Directory**: `/mnt/data/postgresql` (custom location on data drive)
- **Purpose**: Project metadata, experimental results, AeroAPI data
- **Configuration**: Optimized for analytical workloads

### Development Tools
- **Remote Access**: SSH server with key-based authentication
- **Code Editor**: VS Code Remote-SSH integration
- **Notebooks**: Jupyter Lab for interactive analysis
- **Version Control**: Git with GitHub integration

---

## 🔗 Network & Remote Development

### Network Configuration
- **Hostname**: `forge` (accessible from local network)
- **SSH Access**: Port 22 with public key authentication
- **Firewall**: UFW configured for secure remote access
- **DNS**: Local network resolution for easy connection

### Remote Development Workflow
- **Primary Client**: Mac Studio ("Mainframe") for development
- **IDE Integration**: VS Code with Remote-SSH extension
- **Notebook Access**: Jupyter Lab tunneled through SSH
- **File Transfer**: Direct SSH/SCP for large files

### Development Experience
- **Latency**: Low-latency connection for responsive coding
- **GPU Access**: Direct CUDA development from remote IDE
- **Resource Monitoring**: Remote system monitoring via SSH
- **Process Management**: tmux/screen for persistent sessions

---

## ⚙️ Project Portfolio

Forge supports multiple concurrent ML research projects:

### Active Projects

#### Project Bonsai (Primary)
- **Tagline**: *Pruning neural networks with precision and discipline.*
- **Summary**: Project Bonsai is a principled framework for neural network pruning that emphasizes careful, iterative reduction of network parameters based on statistical sensitivity, rather than blunt heuristic approaches (e.g., magnitude pruning). It embodies the philosophy that pruning should be deliberate and thoughtful—akin to tending a bonsai tree—optimizing models while preserving their structure and function. Bonsai focuses on centralized configuration discipline, reproducibility, and rigorous benchmarking across architectures and datasets.
- **Status**: Active development
- **Key Innovation**: Wilcoxon + FANIM breakthrough approach
- **Data Volume**: ~100GB of sensitivity analysis data

#### Project Rodin
- **Tagline**: *Toward concept space reasoning in language models.*
- **Summary**: Project Rodin explores a novel architecture for language understanding, introducing a latent concept space for AI reasoning. Its core components include: **Polly (the Parrot)** - an autoencoder that maps language to latent concept vectors, and **The Difference Engine** - a transformer operating directly in concept space, enabling manipulation of abstract ideas independent of language syntax. The project aims to move beyond associative prediction toward AI systems that can "think" conceptually and semantically—*sculpting meaning like Rodin sculpted marble.*
- **Status**: Research phase
- **Integration**: Shares Forge infrastructure

#### Project Marin  
- **Tagline**: *AI-native assistant for modern inboxes and workflow.*
- **Summary**: Project Marin is an integrated AI system designed to rethink how users interact with email, documents, and schedules. It enables natural language querying, smart classification (e.g., receipts, spam, urgent communications), and workflow automation (saving attachments, generating summaries, etc.). At its core is a foundation model fine-tuned on personal communications data (e.g., emails), combined with contextual LLM-powered orchestration to replace traditional inbox interfaces.
- **Status**: Development
- **Resources**: Utilizes shared datasets and GPU resources

#### Project LACUNA
- **Tagline**: *LLM-Augmented Classification of UNobserved Assumptions.*
- **Summary**: Project LACUNA addresses a long-standing blind spot in biostatistics: the undocumented, intuitive assumption that missing data are Missing At Random (MAR). LACUNA provides an AI-augmented diagnostic framework that integrates classical statistical tests (e.g., Little's MCAR test), analyzes observed missingness patterns, and leverages large language models to incorporate contextual domain knowledge and study metadata, outputting a structured plausibility assessment of MCAR/MAR/MNAR status. It is *not* an imputer—it's a principled decision-support tool that restores rigor to missingness mechanism assessment, embodying the philosophy that *statistics is the quantification of uncertainty—even where certainty is impossible.*
- **Status**: Research
- **Focus**: Biostatistics and missing data analysis

#### Project Epistrophe
- **Tagline**: *Understanding human motivation.*
- **Summary**: Project Epistrophe is a framework for modeling and interpreting core motivational archetypes, reflecting six irreducible psychological drives: Truth Seeker, Caretaker, Engineer, Founder, Creator, and Spirited Competitor. The project is designed to quantify individuals' archetype "stacks," provide archetype-based recommendations (e.g., games, narratives), and offer a motivational profile orthogonal to personality trait models like the Big Five.
- **Status**: Research
- **Requirements**: Psychology-focused ML workloads

#### Project Simulacrum
- **Tagline**: *AI that emulates the style of human masters.*
- **Summary**: Project Simulacrum is an AI framework for style emulation, initially conceived around chess but extensible to other domains of human performance. Its core idea is to train models not merely to optimize for winning or accuracy but to replicate the style, decision patterns, and "feel" of specific individuals (e.g., famous grandmasters). In its chess incarnation, it uses supervised learning from historical game datasets to produce agents that "play like" Tal, Fischer, or Kasparov—even if their strategies aren't optimal by modern engines' standards.
- **Status**: Experimental
- **Architecture**: Leverages Forge's computational capabilities

### Project Isolation Strategy
- **Data Separation**: Each project has dedicated `/mnt/data/{project}/` directory
- **Environment Management**: Conda environments for dependency isolation
- **Resource Allocation**: GPU scheduling for concurrent training
- **Configuration**: Centralized config management via `forge_config.py`

---

## 🏗️ System Design Philosophy

### Performance-First Architecture
- **I/O Optimization**: Separate drives for code, data, and artifacts
- **Memory Hierarchy**: 64GB RAM minimizes swapping during large model training
- **GPU Utilization**: RTX 5070 Ti optimized for research workloads
- **Network**: Gigabit ethernet for fast remote development

### Reliability & Reproducibility
- **Configuration Management**: Centralized path management via ForgeConfig
- **Version Control**: All research code tracked in Git repositories
- **Environment Isolation**: Conda environments prevent dependency conflicts
- **Backup Strategy**: Critical code mirrored to GitHub, data backed up separately

### Scalability Considerations
- **Storage Expansion**: Additional NVMe slots available
- **Memory Upgrades**: Motherboard supports up to 128GB DDR5
- **GPU Upgrades**: PCIe Gen4 x16 slot for future graphics cards
- **Network**: Preparation for high-speed networking if needed

---

## 🔧 Operational Procedures

### System Monitoring
- **Resource Usage**: Regular monitoring of CPU, GPU, memory, and storage
- **Disk Health**: NVMe health monitoring and wear leveling
- **Temperature**: Thermal monitoring for sustained ML workloads
- **Network**: Connection stability for remote development

### Maintenance Schedule
- **Daily**: Automated system updates and security patches
- **Weekly**: Disk usage review and cleanup of temporary files
- **Monthly**: Performance benchmarking and optimization review
- **Quarterly**: Hardware health assessment and environment updates

### Backup Strategy
- **Code**: Continuous sync to GitHub repositories
- **Configuration**: Config files backed up to version control
- **Data**: Critical datasets and models backed up to external storage
- **Documentation**: Project documentation maintained in repositories

---

## 📊 Performance Metrics

### Computational Benchmarks
- **Training Speed**: Optimized for mini-batch SGD with 32-64 sample batches
- **Memory Efficiency**: 16GB VRAM supports models up to ~50M parameters
- **I/O Throughput**: NVMe drives handle high-frequency data loading
- **Network Latency**: <1ms for local network remote development

### Research Productivity
- **Environment Setup**: <5 minutes for new project initialization
- **Experiment Iteration**: Rapid prototyping and testing cycles
- **Data Processing**: FANIM computation at 10-100x speed improvement
- **Model Training**: Concurrent training across multiple projects

---

## 🚀 Future Enhancements

### Hardware Roadmap
- **Storage**: Consider 4TB NVMe upgrade for larger datasets
- **Memory**: Potential 128GB upgrade for transformer-scale models
- **GPU**: Future RTX 6000 series consideration for larger models
- **Networking**: 10GbE upgrade for multi-node training clusters

### Software Evolution
- **Container Strategy**: Docker/Podman for enhanced isolation
- **Orchestration**: Kubernetes for multi-project resource management
- **Monitoring**: Enhanced system monitoring and alerting
- **Automation**: Further automation of experimental workflows

---

## 📝 Important Conventions

### Data Organization Principles
1. **Project-First Structure**: Organize by project, not dataset
2. **Symlink Strategy**: Shared datasets referenced via symlinks
3. **Path Consistency**: Use ForgeConfig for all path management
4. **Naming Convention**: Clear, descriptive directory and file names

### Development Practices
1. **Environment Isolation**: One conda environment per major project
2. **Version Control**: All code committed to Git before experimentation
3. **Configuration**: Experiment configs saved with results for reproducibility
4. **Documentation**: README files in every major directory

### Resource Management
1. **GPU Sharing**: Coordinate GPU usage across concurrent projects
2. **Storage Monitoring**: Regular cleanup of temporary and intermediate files
3. **Process Management**: Use tmux/screen for long-running experiments
4. **Remote Access**: Maintain SSH key security and access logging

---

*This document serves as the definitive reference for Forge system architecture and operational procedures. Updates should be made as the system evolves to support expanding ML research requirements.*