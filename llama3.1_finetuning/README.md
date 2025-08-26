<div id="top">

<!-- HEADER STYLE: MODERN -->
<div align="left" style="position: relative; width: 100%; height: 100%; ">

<img src="readmeai/assets/logos/aurora.svg" width="30%" style="position: absolute; top: 0; right: 0;" alt="Project Logo"/>

# <code>❯ Llama 3.1 Fine-tuning for Educational AI</code>

<em>Fine-tune Llama 3.1 models on educational datasets for Assessment for Learning (AfL) applications.</em>

<!-- BADGES -->
<code>❯ Educational AI • Machine Learning • LLM Fine-tuning</code>

<em>Built with the tools and technologies:</em>

<a href="https://skillicons.dev">
		<img src="https://skillicons.dev/icons?i=md,py">
	</a>
</div>
</div>
<br clear="right">

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
    - [Project Index](#project-index)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Testing](#testing)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

**Project Name: Llama 3.1 Fine-tuning for Educational AI**

**Why Llama 3.1 Fine-tuning for Educational AI?**

This project provides a comprehensive toolkit for fine-tuning Meta's Llama 3.1 language model on educational datasets, specifically focused on Assessment for Learning (AfL) applications. The project enables researchers and educators to create specialized AI assistants for educational contexts.

**Key Features:**

- **🔧 Interactive Jupyter Notebook:** User-friendly interface for model fine-tuning with Unsloth optimization
- **🚀 Data Preprocessing Pipeline:** Convert educational datasets to chat format for model training
- **💻 SLURM Integration:** Scalable training on HPC clusters with Singularity containers
- **� Educational Focus:** Specialized on Assessment for Learning and pedagogical applications
- **🎯 LoRA Fine-tuning:** Efficient parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation)

---

## Features

|      | Component       | Details                              |
| :--- | :-------------- | :----------------------------------- |
| ⚙️  | **Architecture**  | <ul><li>Follows a **modular** design with separate components for data preparation, training, and inference</li><li>Utilizes **parameter-efficient** fine-tuning with LoRA</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>Well-documented Python scripts with type hints</li><li>Jupyter notebooks for interactive development and experimentation</li></ul> |
| 📄 | **Documentation** | <ul><li>Comprehensive **README** with usage instructions</li><li>Inline code comments explaining **educational dataset formatting**</li></ul> |
| 🔌 | **Integrations**  | <ul><li>Integration with **Hugging Face Transformers** and **Unsloth**</li><li>**SLURM** support for HPC cluster training</li></ul> |
| 🧩 | **Modularity**    | <ul><li>**Separate modules** for dataset conversion, training, and inference</li><li>**Reusable components** for different educational datasets</li></ul> |
| 🧪 | **Testing**       | <ul><li>**Test notebooks** for validating dataset formatting</li><li>**Example usage** notebooks demonstrating the workflow</li></ul> |
| ⚡️  | **Performance**   | <ul><li>**4-bit and 8-bit quantization** for memory efficiency</li><li>**LoRA fine-tuning** for reduced computational requirements</li></ul> |
| 🛡️ | **Security**      | <ul><li>**Token-based authentication** for Hugging Face models</li><li>**Environment isolation** using Singularity containers</li></ul> |
| 📦 | **Dependencies**  | <ul><li>Uses **pip** for Python package management</li><li>**Singularity containers** for reproducible HPC environments</li></ul> |

---

## Project Structure

```sh
└── llama3.1_finetuning/
    ├── dataset_preparation/           # Data preprocessing and conversion tools
    │   ├── dataset_converter.py       # Convert DeLorenzi datasets to chat format
    │   ├── converted_datasets/        # Processed datasets in JSON format
    │   │   ├── final_train_set.json  # Training dataset
    │   │   └── final_dev_set.json    # Development/validation dataset
    │   └── delorenzi_datasets/        # Original text format datasets
    │       ├── final_train_set.txt   # Raw training data
    │       └── final_dev_set.txt     # Raw validation data
    ├── training_files/                # Training scripts and configuration
    │   ├── training_script_Base.py   # Main training script with LoRA
    │   ├── script.sh                 # Bash script for SLURM execution
    │   ├── job.slurm                 # SLURM job configuration
    │   └── env.def                   # Singularity environment definition
    ├── finetuning_notebook.ipynb     # Interactive Jupyter notebook for training
    ├── test.ipynb                    # Dataset testing and validation
    └── README.md                     # Project documentation
```

### Project Index

<details open>
	<summary><b><code>/</code></b></summary>
	<!-- __root__ Submodule -->
	<details>
		<summary><b>__root__</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ __root__</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/finetuning_notebook.ipynb'>finetuning_notebook.ipynb</a></b></td>
					<td style='padding: 8px;'>- Interactive Jupyter notebook for fine-tuning Llama 3.1 models using Unsloth optimization<br>- Provides a user-friendly interface for model configuration, dataset loading, and training<br>- Includes LoRA (Low-Rank Adaptation) setup for parameter-efficient fine-tuning<br>- Supports 4-bit quantization for memory-efficient training on educational datasets focused on Assessment for Learning</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/test.ipynb'>test.ipynb</a></b></td>
					<td style='padding: 8px;'>- Dataset testing and validation notebook for educational conversation data<br>- Formats conversation data into proper prompt structures for Llama 3.1 training<br>- Tests dataset loading and preprocessing pipeline<br>- Validates chat format conversion from original DeLorenzi datasets for Assessment for Learning applications</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- training_files Submodule -->
	<details>
		<summary><b>training_files</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ training_files</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/training_files/training_script_Base.py'>training_script_Base.py</a></b></td>
					<td style='padding: 8px;'>- Main training script for Llama 3.1 fine-tuning using LoRA and 8-bit quantization<br>- Implements causal language modeling with SFT (Supervised Fine-Tuning) on educational datasets<br>- Supports command-line arguments for flexible training configuration<br>- Uses completion-only data collation for assistant response training on Assessment for Learning conversations</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/training_files/script.sh'>script.sh</a></b></td>
					<td style='padding: 8px;'>- Bash execution script for running Llama 3.1 fine-tuning on HPC clusters<br>- Configures Singularity container environment and paths for training data<br>- Sets up Hugging Face authentication and model parameters<br>- Launches the training process with specified datasets and output directories</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/training_files/job.slurm'>job.slurm</a></b></td>
					<td style='padding: 8px;'>- SLURM job configuration file for HPC cluster training execution<br>- Specifies resource requirements including GPU allocation and memory<br>- Sets up job scheduling parameters, logging, and email notifications<br>- Configures runtime environment for Llama 3.1 model training with educational datasets</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/training_files/env.def'>env.def</a></b></td>
					<td style='padding: 8px;'>- Singularity container definition file for reproducible training environments<br>- Installs Python dependencies including transformers, datasets, and PEFT libraries<br>- Sets up CUDA environment for GPU-accelerated training<br>- Ensures consistent package versions across different HPC systems for Llama 3.1 fine-tuning</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- dataset_preparation Submodule -->
	<details>
		<summary><b>dataset_preparation</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ dataset_preparation</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/dataset_preparation/dataset_converter.py'>dataset_converter.py</a></b></td>
					<td style='padding: 8px;'>- Dataset conversion utility for transforming DeLorenzi educational datasets<br>- Converts original text format to Hugging Face compatible chat format<br>- Structures conversations as role-content dictionaries for model training<br>- Processes Assessment for Learning dialogues into standardized JSON format for fine-tuning</td>
				</tr>
			</table>
			<!-- converted_datasets Submodule -->
			<details>
				<summary><b>converted_datasets</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ dataset_preparation.converted_datasets</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='/dataset_preparation/converted_datasets/final_train_set.json'>final_train_set.json</a></b></td>
							<td style='padding: 8px;'>- Training dataset containing educational conversations in chat format for Llama 3.1 fine-tuning<br>- Features user-assistant dialogues focused on Assessment for Learning topics and pedagogical practices<br>- Structured as conversation pairs with role-based message formatting<br>- Covers topics like formative assessment, educational barriers, and teaching methodologies in higher education</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='/dataset_preparation/converted_datasets/final_dev_set.json'>final_dev_set.json</a></b></td>
							<td style='padding: 8px;'>- Development/validation dataset for model evaluation during training<br>- Contains educational dialogues in the same format as training data<br>- Used for monitoring model performance and preventing overfitting<br>- Features diverse Assessment for Learning conversations for comprehensive model validation</td>
						</tr>
					</table>
				</blockquote>
			</details>
			<!-- delorenzi_datasets Submodule -->
			<details>
				<summary><b>delorenzi_datasets</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ dataset_preparation.delorenzi_datasets</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='/dataset_preparation/delorenzi_datasets/final_train_set.txt'>final_train_set.txt</a></b></td>
							<td style='padding: 8px;'>- Raw training dataset in original DeLorenzi text format<br>- Contains educational conversations and Q&A pairs about Assessment for Learning<br>- Focuses on challenges and barriers to scaling up AfL in higher education<br>- Source material for conversion to JSON chat format used in model training</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='/dataset_preparation/delorenzi_datasets/final_dev_set.txt'>final_dev_set.txt</a></b></td>
							<td style='padding: 8px;'>- Raw validation dataset in original text format<br>- Contains questions and answers about formative assessment practices<br>- Covers topics like self-assessment benefits, task complexity, and feedback provision<br>- Used as source for generating the development set in JSON format</td>
						</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
</details>

---

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python 3.8+
- **Key Libraries:** 
  - PyTorch
  - Transformers (Hugging Face)
  - PEFT (Parameter-Efficient Fine-Tuning)
  - TRL (Transformer Reinforcement Learning)
  - Unsloth (for optimized training)
  - Datasets
- **Hardware:** NVIDIA GPU with CUDA support (recommended)
- **Environment:** Singularity/Apptainer (for HPC deployment)

### Installation

Build from source and install dependencies:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/gp-1108/NLP_DPO-Finetuning.git
    cd NLP_DPO-Finetuning/llama3.1_finetuning
    ```

2. **Set up Python environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install the dependencies:**

    ```bash
    pip install torch transformers datasets peft trl bitsandbytes
    pip install unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git
    ```

4. **Set up Hugging Face token:**

    ```bash
    export HF_TOKEN="your_huggingface_token_here"
    ```

### Usage

Run the project with different approaches:

**Option 1: Interactive Jupyter Notebook (Recommended for beginners)**

```bash
jupyter notebook finetuning_notebook.ipynb
```

**Option 2: Command Line Training**

```bash
python training_files/training_script_Base.py \
    --ds_train dataset_preparation/converted_datasets/final_train_set.json \
    --ds_dev dataset_preparation/converted_datasets/final_dev_set.json \
    --run_name llama3.1_SFT_educational \
    --output_dir ./output \
    --load_8_bit True \
    --max_seq_len 4096
```

**Option 3: SLURM Cluster Execution**

```bash
cd training_files
sbatch job.slurm
```

**Data Preparation:**

Convert your own educational datasets:

```bash
python dataset_preparation/dataset_converter.py \
    --input_file your_dataset.txt \
    --output_file converted_dataset.json
```

### Testing

Test the project components:

**Dataset Validation:**

```bash
jupyter notebook test.ipynb
```

**Verify Installation:**

```bash
python -c "import torch; import transformers; import datasets; print('All dependencies installed successfully')"
```

**Test Data Conversion:**

```bash
python dataset_preparation/dataset_converter.py --help
```

---

## Roadmap

- [X] **`Dataset Preparation`**: <strike>Implement dataset conversion from DeLorenzi format to chat format.</strike>
- [X] **`LoRA Fine-tuning`**: <strike>Implement parameter-efficient fine-tuning with LoRA.</strike>
- [X] **`HPC Integration`**: <strike>Add SLURM support for cluster-based training.</strike>
- [X] **`Interactive Notebook`**: <strike>Create user-friendly Jupyter notebook interface.</strike>
- [ ] **`Model Evaluation`**: Add comprehensive evaluation metrics for educational AI.
- [ ] **`Multi-GPU Support`**: Implement distributed training capabilities.
- [ ] **`Docker Support`**: Add containerized deployment options.
- [ ] **`Model Serving`**: Implement inference API for deployed models.

---

## Contributing

- **💬 [Join the Discussions](https://github.com/gp-1108/NLP_DPO-Finetuning/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/gp-1108/NLP_DPO-Finetuning/issues)**: Submit bugs found or log feature requests for the `Llama 3.1 Fine-tuning` project.
- **💡 [Submit Pull Requests](https://github.com/gp-1108/NLP_DPO-Finetuning/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your GitHub account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```bash
   git clone https://github.com/YOUR_USERNAME/NLP_DPO-Finetuning.git
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```bash
   git checkout -b feature/educational-ai-enhancement
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```bash
   git commit -m 'Add support for new educational dataset format'
   ```
6. **Push to GitHub**: Push the changes to your forked repository.
   ```bash
   git push origin feature/educational-ai-enhancement
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com/gp-1108/NLP_DPO-Finetuning/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=gp-1108/NLP_DPO-Finetuning">
   </a>
</p>
</details>

---

## License

This project is protected under the [MIT](https://choosealicense.com/licenses/mit/) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/mit/) file.

---

## Acknowledgments

- **Unsloth AI** for providing optimized fine-tuning capabilities
- **Hugging Face** for the transformers library and model hosting
- **Meta AI** for the Llama 3.1 base model
- **DeLorenzi et al.** for the educational assessment datasets
- **Assessment for Learning (AfL)** research community for educational insights

<div align="right">

[![][back-to-top]](#top)

</div>


[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square


---
