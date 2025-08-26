<p align="center">
    <img src="https://img.icons8.com/?size=512&id=55494&format=png" align="center" width="30%">
</p>
<p align="center"><h1 align="center"><code>❯ LLAMA3.1-DPO</code></h1></p>
<p align="center">
	<em>Empowering models, enhancing conversations, shaping interactions through Direct Preference Optimization.</em>
</p>
<p align="center">
	<!-- Shields.io badges disabled, using skill icons. --></p>
<p align="center">Built with the tools and technologies:</p>
<p align="center">
	<a href="https://skillicons.dev">
		<img src="https://skillicons.dev/icons?i=python,pytorch,docker,linux">
	</a></p>
<br>

## 🔗 Table of Contents

- [📍 Overview](#-overview)
- [👾 Features](#-features)
- [📁 Project Structure](#-project-structure)
  - [📂 Project Index](#-project-index)
- [🚀 Getting Started](#-getting-started)
  - [☑️ Prerequisites](#-prerequisites)
  - [⚙️ Installation](#-installation)
  - [🤖 Usage](#🤖-usage)
  - [🧪 Testing](#🧪-testing)
- [📌 Project Roadmap](#-project-roadmap)
- [🔰 Contributing](#-contributing)
- [🎗 License](#-license)
- [🙌 Acknowledgments](#-acknowledgments)

---

## 📍 Overview

This project implements Direct Preference Optimization (DPO) fine-tuning for LLaMA 3.1 models, specifically designed for educational dialogue applications. The system processes pedagogical dialogue data to create preference pairs and fine-tune models using the DPO methodology, improving conversational quality through human preference learning.

Key capabilities include:
- **DPO Fine-tuning**: Advanced preference-based training using the TRL library
- **Educational Dialogue Processing**: Specialized components for handling pedagogical conversations
- **Distributed Training**: Support for multi-GPU training with SLURM integration  
- **Interactive Inference**: CLI chatbot interface for testing fine-tuned models
- **Automated Pipeline**: End-to-end workflow from data processing to model deployment

---

## 👾 Features

|      | Feature         | Summary       |
| :--- | :---:           | :---          |
| ⚙️  | **Architecture**  | <ul><li>Utilizes **PEFT** and **DPOTrainer** for distributed training</li><li>Configures **Lora settings** for model training</li><li>Employs **transformers** and **threading** for efficient processing</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>Structured logging with **rotating file logger** for easy debugging</li><li>Utilizes **pedagogical rules** for dialogue transformation</li><li>Follows **PEP8** coding standards for consistency</li></ul> |
| 📄 | **Documentation** | <ul><li>Comprehensive documentation in **Python** with **21 Python files**</li><li>Includes **testing notebook** for validating components</li><li>**SLURM job scripts** for job configurations</li></ul> |
| 🔌 | **Integrations**  | <ul><li>Integration with **wandb** for logging training progress</li><li>Utilizes **OpenAI's API** for generating educational dialogues</li><li>**CUDA 12.4** for GPU acceleration</li></ul> |
| 🧩 | **Modularity**    | <ul><li>Components like **Dialogue**, **Chunk**, and **Document** for structured data management</li><li>**BaseComponent** class for extensibility</li><li>**BaseLoader** for consistent dataset loading</li></ul> |
| 🧪 | **Testing**       | <ul><li>Includes **testing notebook** for validating code functionality</li><li>**Unit tests** for critical components</li><li>**Verbose mode** for insights during chatbot interactions</li></ul> |
| ⚡️  | **Performance**   | <ul><li>**Mixed precision** and **distributed training** for faster model training</li><li>**GPU-accelerated inference** for quick response generation</li><li>**Efficient dialogue data transformation** for optimization</li></ul> |
| 🛡️ | **Security**      | <ul><li>Secure environment setup with **CUDA 12.4** and **PyTorch** compatibility</li><li>**Structured logging** for monitoring system behavior</li><li>**API key usage** for accessing external services securely</li></ul> |
| 📦 | **Dependencies**  | <ul><li>Includes essential dependencies like **Python modules** and **CUDA 12.4**</li><li>**PEFT model configuration** and **tokenizer** for training</li><li>**Hugging Face Transformers** for NLP tasks</li></ul> |

---

## 📁 Project Structure

```sh
└── /
    ├── __pycache__
    │   └── utils.cpython-312.pyc
    ├── core
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── components
    │   ├── loaders
    │   ├── logger.py
    │   └── processes
    ├── dpo_finetuning.py
    ├── env.def
    ├── inference.py
    ├── inference.sh
    ├── negative_ans_from_sft.py
    ├── script.sh
    ├── slurm_creator.sh
    └── utils.py
```


### 📂 Project Index
<details open>
	<summary><b><code>/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='/script.sh'>script.sh</a></b></td>
				<td>- Configure environment variables and execute a script for fine-tuning a model on a dataset, utilizing mixed precision and distributed training<br>- The script launches the training process with specified parameters and settings, leveraging the provided environment setup for efficient execution.</td>
			</tr>
			<tr>
				<td><b><a href='/inference.py'>inference.py</a></b></td>
				<td>- Facilitates an interactive CLI chatbot using a pre-trained language model for generating responses<br>- The code orchestrates the chatbot's functionality, including formatting conversation prompts, streaming response generation, and managing user interactions<br>- It leverages transformers and threading for efficient processing, offering a seamless conversational experience with the assistant<br>- The chatbot supports customizable response parameters and provides insights through verbose mode.</td>
			</tr>
			<tr>
				<td><b><a href='/dpo_finetuning.py'>dpo_finetuning.py</a></b></td>
				<td>- Fine-tunes a model using PEFT and DPOTrainer, leveraging distributed training and wandb logging<br>- Loads PEFT model configuration, base model, tokenizer, and dataset<br>- Configures training parameters and Lora settings<br>- Initializes DPO trainer and commences model training.</td>
			</tr>
			<tr>
				<td><b><a href='/negative_ans_from_sft.py'>negative_ans_from_sft.py</a></b></td>
				<td>- Generates negative answers using an SFT model by processing datasets and model inputs<br>- The code loads datasets, formats prompts, and utilizes the SFT model to generate negative responses based on given prompts and positive answers<br>- The results are saved in an output JSONL file, providing a structured approach to generating negative responses for conversational interactions.</td>
			</tr>
			<tr>
				<td><b><a href='/env.def'>env.def</a></b></td>
				<td>- Configure the project environment by installing necessary dependencies, including Python modules, CUDA 12.4, and setting up paths<br>- This file sets up the required software components for the project to run smoothly on an Ubuntu 22.04 base, ensuring compatibility with libraries like PyTorch and Hugging Face Transformers.</td>
			</tr>
			<tr>
				<td><b><a href='/slurm_creator.sh'>slurm_creator.sh</a></b></td>
				<td>- Generates SLURM job files with unique configurations for DPO fine-tuning experiments<br>- Constructs job names, directories, and commands based on hyperparameters<br>- Utilizes SLURM for job submission with specified resource allocations<br>- Facilitates parallel execution of multiple jobs with varying configurations.</td>
			</tr>
			<tr>
				<td><b><a href='/inference.sh'>inference.sh</a></b></td>
				<td>- Facilitates GPU-accelerated model inference by setting environment variables and executing a Python script<br>- The script takes model path, temperature, and max tokens as inputs for generating new text<br>- This file plays a crucial role in orchestrating the inference process within the project architecture.</td>
			</tr>
			<tr>
				<td><b><a href='/utils.py'>utils.py</a></b></td>
				<td>- The code file `utils.py` facilitates the transformation of dialogue data into a standardized dataset format suitable for training models<br>- It includes functions for formatting interactions, converting data from a custom loader, filtering datasets based on length and statistical criteria, loading a specific dataset, and merging multiple datasets efficiently<br>- These operations are crucial for preparing high-quality training data for natural language processing tasks.</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- core Submodule -->
		<summary><b>core</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='/core/logger.py'>logger.py</a></b></td>
				<td>- Configures a rotating file logger with specified settings, enabling structured logging to a file<br>- The logger setup includes log level, file rotation size, and backup log files count<br>- The code initializes the logger with default or custom settings, facilitating easy integration for logging messages at different levels.</td>
			</tr>
			</table>
			<details>
				<summary><b>processes</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='/core/processes/DPOGenerator.py'>DPOGenerator.py</a></b></td>
						<td>- Generates Direct Preference Optimization (DPO) training data by applying pedagogical rules to transform original dialogue turns<br>- Handles loading dialogues, rule application, and saving generated preference pairs<br>- Implements a depth-first search approach to create dialogue variations and ensures only high-scoring rules are applied<br>- The generated data aids in training models for dialogue optimization.</td>
					</tr>
					<tr>
						<td><b><a href='/core/processes/DialogueGenerator.py'>DialogueGenerator.py</a></b></td>
						<td>- Generates educational dialogues between a student and a tutor by processing text documents, sending chunks to OpenAI's API, and saving dialogues in JSONL format<br>- Handles text chunks, generates dialogues, and ensures coherence<br>- Utilizes OpenAI's API, requires API key, and processes input files containing coherent text chunks<br>- Saves dialogues between student and tutor based on content.</td>
					</tr>
					<tr>
						<td><b><a href='/core/processes/ChunkExtractor.py'>ChunkExtractor.py</a></b></td>
						<td>- The ChunkExtractor class processes PDF files, extracting text into structured chunks for readability<br>- It preprocesses text, handles special content like emails and URLs, and saves results in JSONL format<br>- The class ensures coherence and readability by maintaining chunk length thresholds and removing unnecessary elements.</td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>components</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='/core/components/Dialogue.py'>Dialogue.py</a></b></td>
						<td>- The Dialogue class manages dialogues composed of multiple turns, handling creation, serialization, and dialogue management<br>- It uniquely identifies dialogues and extracts chunk IDs<br>- The class provides methods to convert dialogues to JSON strings and load dialogues from JSON.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/Chunk.py'>Chunk.py</a></b></td>
						<td>- Manages text chunks with unique identifiers, providing creation, serialization, and string representation functionality<br>- Handles chunk ID extraction and construction, JSON serialization, and initialization from JSON string.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/PedagogicalRules.py'>PedagogicalRules.py</a></b></td>
						<td>- Handles pedagogical rules loaded from a text file, providing bidirectional mapping between rule indices and texts<br>- Supports rule retrieval by index or text and enables iteration over rules<br>- The class facilitates efficient management and access to pedagogical rules within the project architecture.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/BaseSubComponent.py'>BaseSubComponent.py</a></b></td>
						<td>Defines a base class for sub-components with methods to convert to/from JSON strings.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/Document.py'>Document.py</a></b></td>
						<td>- Manages document data, including text chunks, file info, and ID<br>- Generates and retrieves chunks, converts data to JSON, and loads data from JSON<br>- Provides a string representation of the document.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/DPODialogue.py'>DPODialogue.py</a></b></td>
						<td>- Manages Direct Preference Optimization (DPO) dialogue data, including ID generation, history tracking, and JSON serialization/deserialization<br>- Provides methods to extract chunk IDs and document ID from dialogue ID<br>- Enables creation and manipulation of DPO dialogues within the project architecture.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/BaseComponent.py'>BaseComponent.py</a></b></td>
						<td>Defines a base class for components in the dataset generation pipeline, offering methods to convert components to JSON strings, create instances from JSON strings, and save JSON representations to a file in JSONL format.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/DPOTurn.py'>DPOTurn.py</a></b></td>
						<td>- Defines a class for managing conversational turns in a Direct Preference Optimization system<br>- Stores student questions, positive/negative answers, and applied rules<br>- Provides methods to convert data to/from JSON and generate string representations of turns<br>- Facilitates structured handling of conversational exchanges within the project architecture.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/Turn.py'>Turn.py</a></b></td>
						<td>- Handles the storage and serialization of conversation turns, capturing user messages and assistant responses<br>- Converts turns to JSON strings and loads them from JSON representations<br>- Provides string representations of turns for easy viewing and debugging within the conversation context.</td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>loaders</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='/core/loaders/DPODialogueLoader.py'>DPODialogueLoader.py</a></b></td>
						<td>- The DPODialogueLoader file facilitates loading and processing of dialogue data from a JSONL file<br>- It creates DPODialogue objects, builds an index, and provides methods to extract unique DPO IDs, retrieve turns by dialogue ID, and access dialogue objects by ID<br>- Additionally, it offers functionality to check for the presence of standard dialogue IDs within the dataset.</td>
					</tr>
					<tr>
						<td><b><a href='/core/loaders/BaseLoader.py'>BaseLoader.py</a></b></td>
						<td>- BaseLoader.py provides a foundational structure for dataset loaders in the project<br>- It defines key behaviors such as loading data, creating an index, and enabling key lookups<br>- This class serves as a blueprint for implementing dataset loaders that read from JSONL files, ensuring consistency and extensibility across different loader implementations.</td>
					</tr>
					<tr>
						<td><b><a href='/core/loaders/DocumentLoader.py'>DocumentLoader.py</a></b></td>
						<td>- The DocumentLoader class in core/loaders/DocumentLoader.py loads and processes data from a JSONL file into Document objects<br>- It creates Document objects from non-empty lines in the file and builds an index mapping document IDs to their positions<br>- It provides methods to retrieve documents by ID and load the index of all document IDs in the dataset.</td>
					</tr>
					<tr>
						<td><b><a href='/core/loaders/DialogueLoader.py'>DialogueLoader.py</a></b></td>
						<td>- Manages loading and accessing dialogues from a JSONL file, creating an index mapping dialogue IDs<br>- Enables retrieval of dialogues by document ID or unique identifier, and provides methods to load dialogue IDs and retrieve all dialogue IDs in the dataset.</td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
</details>

---
## 🚀 Getting Started

### ☑️ Prerequisites

Before getting started with , ensure your runtime environment meets the following requirements:

- **Programming Language:** Python


### ⚙️ Installation

Install LLAMA3.1-DPO using one of the following methods:

**Build from source:**

1. Clone the repository:
```sh
❯ git clone https://github.com/gp-1108/NLP_DPO-Finetuning.git
❯ cd NLP_DPO-Finetuning/llama3.1_dpo
```

2. Set up the Apptainer/Singularity environment:
```sh
❯ apptainer build env_cuda.sif env.def
```

3. Configure your environment variables:
```sh
# Edit script.sh and inference.sh with your tokens and paths
export HF_TOKEN="your_huggingface_token"
export WANDB_API_KEY="your_wandb_token"
export CUDA_HOME="/usr/local/cuda-12.4"
```



### 🤖 Usage

#### DPO Fine-tuning
Train a model using Direct Preference Optimization:
```sh
❯ ./script.sh
```

#### Interactive Inference
Run the chatbot interface with a fine-tuned model:
```sh
❯ ./inference.sh /path/to/model 0.7 512
```

#### Generate SLURM Jobs
Create multiple training jobs with different hyperparameters:
```sh
❯ ./slurm_creator.sh
```

#### Generate Negative Responses
Create negative responses from an SFT model for DPO training:
```sh
❯ python negative_ans_from_sft.py --model_path /path/to/sft/model --dataset_path /path/to/dataset.jsonl
```

### 🧪 Testing
Run the test notebook to validate the training pipeline:
```sh
❯ jupyter notebook test.ipynb
```

---
## 📌 Project Roadmap

- [X] **`DPO Implementation`**: <strike>Core DPO fine-tuning functionality with PEFT integration.</strike>
- [X] **`Interactive Inference`**: <strike>CLI chatbot interface for model testing.</strike>
- [X] **`SLURM Integration`**: <strike>Automated job generation for distributed training.</strike>
- [ ] **`Model Evaluation`**: Comprehensive evaluation metrics and benchmarking.
- [ ] **`Web Interface`**: Browser-based interface for easier model interaction.
- [ ] **`Multi-modal Support`**: Extension to handle image and text inputs.

---

## 🔰 Contributing

- **💬 [Join the Discussions](https://github.com/gp-1108/NLP_DPO-Finetuning/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/gp-1108/NLP_DPO-Finetuning/issues)**: Submit bugs found or log feature requests for the `LLAMA3.1-DPO` project.
- **💡 [Submit Pull Requests](https://github.com/gp-1108/NLP_DPO-Finetuning/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your GitHub account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/gp-1108/NLP_DPO-Finetuning.git
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to GitHub**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
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

## 🎗 License

This project is protected under the MIT License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/mit/) file.

---

## 🙌 Acknowledgments

- **Hugging Face Transformers** and **TRL** libraries for DPO implementation
- **PEFT** for parameter-efficient fine-tuning capabilities  
- **Weights & Biases** for experiment tracking and logging
- **Meta AI** for the LLaMA 3.1 model architecture
- **OpenAI** for dialogue generation capabilities used in data preprocessing

---
