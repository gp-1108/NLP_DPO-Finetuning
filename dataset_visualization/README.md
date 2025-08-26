<p align="center">
    <img src="https://img.icons8.com/?size=512&id=55494&format=png" align="center" width="30%">
</p>
<p align="center"><h1 align="center"><code>❯ Dataset Visualization</code></h1></p>
<p align="center">
	<em>Interactive web interface for visualizing educational dialogue datasets and DPO training data.</em>
</p>
<p align="center">
	<!-- Shields.io badges disabled, using skill icons. --></p>
<p align="center">Built with the tools and technologies:</p>
<p align="center">
	<a href="https://skillicons.dev">
		<img src="https://skillicons.dev/icons?i=docker,html,flask,python">
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

Dataset Visualization is a Flask-based web application designed to provide an interactive interface for exploring educational dialogue datasets. This tool allows researchers and developers to browse through documents, text chunks, generated dialogues, and Direct Preference Optimization (DPO) training data in an intuitive web interface. The application supports visualization of pedagogical dialogue data, making it easier to understand and analyze the structure of educational conversations and training datasets.

---

## 👾 Features

|      | Feature         | Summary       |
| :--- | :---:           | :---          |
| ⚙️  | **Architecture**  | <ul><li>Utilizes **Flask** for web application development.</li><li>Follows a **modular** design with components like `Dialogue`, `Chunk`, and `Document`.</li><li>Implements **Docker** containerization for deployment.</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>Consistent use of **Python 3.9** for codebase development.</li><li>Employs **pydantic** for data validation and settings management.</li><li>Includes a **rotating file logger** for efficient log management.</li></ul> |
| 📄 | **Documentation** | <ul><li>Comprehensive **Python** documentation with a mix of **txt**, **py**, and **html** files.</li><li>Utilizes **Flask** templating for dynamic content generation.</li><li>Includes detailed **installation** and **usage commands** for **pip** and **docker**.</li></ul> |
| 🔌 | **Integrations**  | <ul><li>Integrates with **OpenAI API** for generating educational dialogues.</li><li>Uses **Flask** routes to handle various data sources and interactions.</li><li>Includes **pedagogical rules** for dialogue generation.</li></ul> |
| 🧩 | **Modularity**    | <ul><li>Organized into components like `Dialogue`, `Chunk`, and `Document` for clear separation of concerns.</li><li>Utilizes **base classes** for sub-components to maintain a structured codebase.</li><li>Follows a **depth-first search** approach for generating DPO training data.</li></ul> |
| 🧪 | **Testing**       | <ul><li>Uses **pytest** for testing the project.</li><li>Includes test commands for running the test suite.</li><li>Ensures compatibility with specified versions of dependencies for consistent testing.</li></ul> |
| ⚡️  | **Performance**   | <ul><li>Optimizes performance by processing text in chunks with overlaps for coherent dialogues.</li><li>Utilizes **Flask** templating for efficient rendering of dynamic content.</li><li>Handles PDF processing efficiently to extract structured text chunks.</li></ul> |
| 🛡️ | **Security**      | <ul><li>Implements **security best practices** for Flask web application development.</li><li>Ensures **data validation** using **pydantic** for secure input handling.</li><li>Follows **structured logging** practices for secure log management.</li></ul> |

---

## 📁 Project Structure

```sh
└── /
    ├── Dockerfile
    ├── app.log
    ├── core
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── components
    │   ├── loaders
    │   ├── logger.py
    │   └── processes
    ├── readme-ai.md
    ├── requirements.txt
    ├── run.py
    ├── static
    │   ├── css
    │   └── js
    └── templates
        ├── chunk.html
        ├── dialogue.html
        ├── document.html
        ├── dpo_dialogue.html
        └── home.html
```


### 📂 Project Index
<details open>
	<summary><b><code>/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='/requirements.txt'>requirements.txt</a></b></td>
				<td>- Facilitates project dependencies management by specifying required packages and versions<br>- This file ensures the project utilizes specific versions of Flask, openai, pydantic, pypdf, tqdm, and Unidecode to maintain compatibility and functionality.</td>
			</tr>
			<tr>
				<td><b><a href='/run.py'>run.py</a></b></td>
				<td>- Enables a Flask web application to render various templates based on data loaded from different sources<br>- Handles routes for displaying documents, chunks, dialogues, and DPO dialogues with associated information<br>- Supports dynamic content generation for user interaction.</td>
			</tr>
			<tr>
				<td><b><a href='/Dockerfile'>Dockerfile</a></b></td>
				<td>Facilitates Docker container setup for a Python project by defining the base image, setting the working directory, copying project files, updating core links, installing dependencies, exposing a port, and running a Flask app.</td>
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
				<td>- Configures a rotating file logger with specified settings, allowing for easy log management and storage<br>- The logger can be set up with custom log levels and file paths, ensuring efficient logging of messages with timestamps and severity levels<br>- The code promotes structured logging practices within the project architecture.</td>
			</tr>
			</table>
			<details>
				<summary><b>processes</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='/core/processes/DPOGenerator.py'>DPOGenerator.py</a></b></td>
						<td>- Generates Direct Preference Optimization (DPO) training data by applying pedagogical rules to transform original dialogue turns<br>- Implements a depth-first search approach to create variations of dialogues, handling loading, rule application, and saving generated preference pairs<br>- The code iterates through dialogues, generates preference data, and logs any encountered errors.</td>
					</tr>
					<tr>
						<td><b><a href='/core/processes/DialogueGenerator.py'>DialogueGenerator.py</a></b></td>
						<td>- Generates educational dialogues between a student and a tutor by processing text documents, sending chunks to OpenAI's API with a specified prompt, and saving dialogues in JSONL format<br>- Handles text in chunks of ~5000 characters with 1000-character overlaps, ensuring coherent dialogues<br>- The class serves as a wrapper around the OpenAI API, creating dialogues based on extracted text from PDF files.</td>
					</tr>
					<tr>
						<td><b><a href='/core/processes/ChunkExtractor.py'>ChunkExtractor.py</a></b></td>
						<td>- The ChunkExtractor class processes PDF files, extracting text into structured chunks<br>- It preprocesses text, removes references, and handles special content like emails and URLs<br>- The class ensures coherence and readability of text chunks, saving results in a JSONL file format<br>- It recursively processes PDFs in a specified directory, avoiding duplicates and maintaining text quality.</td>
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
						<td>- The Dialogue class manages dialogues composed of multiple turns, handling creation, serialization, and dialogue management<br>- It uniquely identifies dialogues and extracts chunk IDs<br>- The class provides methods to convert dialogues to JSON strings and initialize from JSON.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/Chunk.py'>Chunk.py</a></b></td>
						<td>- Manages text chunks with unique identifiers, providing creation, serialization, and string representation functionality<br>- Handles chunk ID extraction and construction, JSON serialization, and initialization from JSON string<br>- Enables efficient handling and manipulation of text segments within the project architecture.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/PedagogicalRules.py'>PedagogicalRules.py</a></b></td>
						<td>- Manages bidirectional mapping between rule indices and texts, facilitating rule retrieval and iteration<br>- Loads rules from a text file, enabling access by index or text<br>- Supports iteration over rules and provides methods for efficient rule management within the codebase architecture.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/BaseSubComponent.py'>BaseSubComponent.py</a></b></td>
						<td>Defines a base class for sub-components with methods to handle JSON serialization/deserialization.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/Document.py'>Document.py</a></b></td>
						<td>- Manages document data, including text chunks, file info, and ID<br>- Converts data to JSON and loads from JSON string<br>- Retrieves specific chunks by ID<br>- Provides document ID generation.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/DPODialogue.py'>DPODialogue.py</a></b></td>
						<td>- Manages Direct Preference Optimization (DPO) dialogue data, including ID generation, history tracking, and JSON serialization/deserialization<br>- Provides methods to extract chunk IDs and document ID from dialogue ID<br>- Enables creation of DPO dialogue chains by generating IDs for previous dialogues.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/BaseComponent.py'>BaseComponent.py</a></b></td>
						<td>Defines a base class for components in the dataset generation pipeline, offering methods to convert components to JSON strings, create instances from JSON strings, and save JSON representations to a file in JSONL format.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/DPOTurn.py'>DPOTurn.py</a></b></td>
						<td>- Defines a class for managing conversational turns in a DPO system<br>- Stores student questions, positive and negative answers, and the rule applied<br>- Provides methods to convert data to/from JSON and generate string representations of the turn<br>- Crucial for managing conversational exchanges within the project's architecture.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/Turn.py'>Turn.py</a></b></td>
						<td>- Handles the storage and serialization of conversation turns, containing user messages and assistant responses<br>- Converts turns to JSON strings and loads them from JSON representations<br>- Provides string representations of turns for easy viewing and debugging within the project's architecture.</td>
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
						<td>- The DPODialogueLoader file facilitates loading and processing of dialogue data from a JSONL file<br>- It creates DPODialogue objects, builds an index, and provides methods to retrieve unique DPO IDs, turns by dialogue ID, and specific dialogues<br>- Additionally, it includes functionality to check for the presence of a standard dialogue ID in the dataset.</td>
					</tr>
					<tr>
						<td><b><a href='/core/loaders/BaseLoader.py'>BaseLoader.py</a></b></td>
						<td>- Defines a foundational class for dataset loaders, enabling key lookups and length queries<br>- Implements abstract methods for loading data and creating an index<br>- Acts as a base for implementing dataset loaders reading from JSONL files, facilitating dataset manipulation and access within the codebase architecture.</td>
					</tr>
					<tr>
						<td><b><a href='/core/loaders/DocumentLoader.py'>DocumentLoader.py</a></b></td>
						<td>- The DocumentLoader class in core/loaders/DocumentLoader.py reads and processes data from a JSONL file into Document objects<br>- It creates Document objects from non-empty lines in the file and builds an index mapping document IDs to their positions in the resulting list<br>- It also provides methods to retrieve documents by ID and load the index of document IDs from the data collection.</td>
					</tr>
					<tr>
						<td><b><a href='/core/loaders/DialogueLoader.py'>DialogueLoader.py</a></b></td>
						<td>- DialogueLoader.py facilitates loading, indexing, and retrieving dialogues from a JSONL file<br>- It creates Dialogue objects from the file, maps dialogue IDs to positions, and provides methods to access dialogues by document ID or unique identifier<br>- The file also supports loading dialogue IDs and retrieving all dialogue IDs in the dataset.</td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<details> <!-- templates Submodule -->
		<summary><b>templates</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='/templates/chunk.html'>chunk.html</a></b></td>
				<td>- Render a detailed chunk view with chunk ID, associated document, and text content<br>- Displayed in an HTML template, it leverages dynamic data to showcase chunk-specific information within the project's architecture.</td>
			</tr>
			<tr>
				<td><b><a href='/templates/home.html'>home.html</a></b></td>
				<td>- Render a dynamic home page displaying categorized documents, dialogues, and DPO dialogues with corresponding links and filter buttons<br>- The page structure includes containers for each category, listing items fetched from the backend<br>- Styling and scripts are linked for enhanced user experience.</td>
			</tr>
			<tr>
				<td><b><a href='/templates/dialogue.html'>dialogue.html</a></b></td>
				<td>- Render a detailed HTML page displaying dialogue information, involved chunks, DPO dialogues, and a chat window with user and assistant interactions<br>- The page includes dynamic content like dialogue ID, document links, chunk IDs, and DPO dialogue IDs<br>- Styling and functionality are enhanced through CSS and JavaScript files linked in the header.</td>
			</tr>
			<tr>
				<td><b><a href='/templates/document.html'>document.html</a></b></td>
				<td>- Generates an HTML document displaying document details, chunks, and raw dialogues<br>- Includes dynamic content rendering using Flask's templating engine<br>- Enhances user experience with interactive elements and styling.</td>
			</tr>
			<tr>
				<td><b><a href='/templates/dpo_dialogue.html'>dpo_dialogue.html</a></b></td>
				<td>- Render a detailed HTML template for displaying DPO Dialogue information, including dialogue ID, related documents, chunks, original dialogue, and chat history with student questions, positive and negative answers, and applied rules<br>- The template also links to specific pages for further details.</td>
			</tr>
			</table>
		</blockquote>
	</details>
</details>

---
## 🚀 Getting Started

### ☑️ Prerequisites

Before getting started with Dataset Visualization, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python 3.9+
- **Package Manager:** Pip
- **Container Runtime:** Docker (optional)

**Data Requirements:**
The application expects the following JSONL files in your data directory:
- `extracted_texts.jsonl` - Document chunks extracted from PDFs
- `dialogues.jsonl` - Generated educational dialogues
- `dpo_dialogues.jsonl` - Direct Preference Optimization training data
- `rules.txt` - Pedagogical rules file (in prompts directory)


### ⚙️ Installation

Install  using one of the following methods:

**Build from source:**

1. Clone the repository:
```sh
❯ git clone https://github.com/gp-1108/NLP_DPO-Finetuning.git
```

2. Navigate to the dataset visualization directory:
```sh
❯ cd NLP_DPO-Finetuning/dataset_visualization
```

3. Install the project dependencies:


**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pip install -r requirements.txt
```


**Using `docker`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Docker-2CA5E0.svg?style={badge_style}&logo=docker&logoColor=white" />](https://www.docker.com/)

```sh
❯ docker build -t dataset-visualization .
```




### 🤖 Usage
Run the Dataset Visualization application using the following methods:

**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ python run.py [data_path] [rules_path]
```

For example:
```sh
❯ python run.py /path/to/dataset_generation/data /path/to/dataset_generation/prompts/rules.txt
```

If no arguments are provided, it defaults to:
```sh
❯ python run.py
# Uses default paths: /home/gp1108/Code/Thesis/dataset_generation/data and /home/gp1108/Code/Thesis/dataset_generation/prompts/rules.txt
```

**Using `docker`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Docker-2CA5E0.svg?style={badge_style}&logo=docker&logoColor=white" />](https://www.docker.com/)

```sh
❯ docker run -p 5005:5005 dataset-visualization
```

The application will be available at `https://localhost:5005` (Note: SSL certificates are required for HTTPS).


### 🧪 Testing
This visualization tool is primarily for exploring datasets. To test the application:

**Check application startup:**
```sh
❯ python run.py --help
```

**Verify dependencies:**
```sh
❯ pip list | grep -E "(Flask|pydantic|pypdf|tqdm|Unidecode|openai)"
```

**Test with sample data:**
Make sure you have the required JSONL files (`extracted_texts.jsonl`, `dialogues.jsonl`, `dpo_dialogues.jsonl`) in your data directory before running the application.


---
## 📌 Project Roadmap

- [X] **`Core Visualization`**: <strike>Implement basic document, chunk, and dialogue visualization.</strike>
- [X] **`DPO Support`**: <strike>Add support for visualizing Direct Preference Optimization dialogues.</strike>
- [X] **`Docker Integration`**: <strike>Containerize the application for easy deployment.</strike>
- [ ] **`Interactive Filtering`**: Enhance filtering capabilities for better dataset exploration.
- [ ] **`Data Export`**: Add functionality to export filtered views and statistics.
- [ ] **`Performance Optimization`**: Improve loading times for large datasets.

---

## 🔰 Contributing

- **💬 [Join the Discussions](https://github.com/gp-1108/NLP_DPO-Finetuning/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/gp-1108/NLP_DPO-Finetuning/issues)**: Submit bugs found or log feature requests for the `Dataset Visualization` project.
- **💡 [Submit Pull Requests](https://github.com/gp-1108/NLP_DPO-Finetuning/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your GitHub account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/your-username/NLP_DPO-Finetuning.git
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

This project is part of a research thesis on NLP and DPO fine-tuning. For more details about usage and licensing, please refer to the main repository.

---

## 🙌 Acknowledgments

- Flask framework for the web application foundation
- OpenAI for dialogue generation capabilities
- Pydantic for robust data validation
- The educational dialogue research community for inspiration and guidance

---
