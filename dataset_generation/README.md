<p align="center">
    <img src="https://img.icons8.com/?size=512&id=55494&format=png" align="center" width="30%">
</p>
<p align="center"><h1 align="center"><code>DPO DATASET GENERATION</code></h1></p>
<p align="center">
	<em>Transforming Text into Engaging Dialogues to train you LLM thorugh RLHF</em>
</p>
<p align="center">
	<!-- Shields.io badges disabled, using skill icons. --></p>
<p align="center">Built with the tools and technologies:</p>
<p align="center">
	<a href="https://skillicons.dev">
		<img src="https://skillicons.dev/icons?i=md">
		<img src="https://skillicons.dev/icons?i=py">
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

---

## 📍 Overview

This project addresses the challenge of generating structured dialogue datasets from PDF documents, enhancing educational interactions between students and tutors. Key features include text extraction, dialogue generation, and pedagogical rule evaluation, promoting personalized learning experiences. Ideal for educators and researchers, it streamlines the creation and analysis of educational dialogues for improved engagement and outcomes.

The main goal is to create a simple yet effective way to convert PDF files into complex dialogues following some pedagogical rules. These dialogues are then well suited for a pletora of applications, amongst all the preferred one is Direct Preference Optimization for LLMs.

---

## 👾 Features

|      | Feature         | Summary       |
| :--- | :---:           | :---          |
| ⚙️  | **Architecture**  | <ul><li>Utilizes a modular architecture that separates concerns across various components, enhancing maintainability.</li><li>Incorporates a robust logging mechanism via `<logger>` for monitoring and troubleshooting.</li><li>Employs a structured approach to dialogue generation and processing, facilitating educational interactions.</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>Follows consistent coding standards across Python files, promoting readability and maintainability.</li><li>Utilizes serialization methods for JSON data interchange, ensuring data integrity.</li><li>Includes comprehensive error handling to manage exceptions effectively.</li></ul> |
| 📄 | **Documentation** | <ul><li>Documentation is primarily in Python, with 17 `.py` files and 2 Jupyter notebooks (`.ipynb`).</li><li>Includes example usage in `example_usage.ipynb` to demonstrate core functionalities.</li><li>Text files (`.txt`) provide essential prompts and rules for dialogue generation.</li></ul> |
| 🔌 | **Integrations**  | <ul><li>Integrates with OpenAI's API for generating educational dialogues, enhancing interaction quality.</li><li>Utilizes visualization libraries in Jupyter notebooks for data analysis and interpretation.</li><li>Supports loading and processing of JSONL files for structured data management.</li></ul> |
| 🧩 | **Modularity**    | <ul><li>Components are designed to be reusable and interchangeable, promoting a clean architecture.</li><li>Defines base classes for loaders and components, facilitating consistent behavior across the codebase.</li><li>Encapsulates dialogue and chunk management within dedicated classes, enhancing clarity.</li></ul> |
| ⚡️  | **Performance**   | <ul><li>Optimizes dialogue generation through efficient data processing and chunk management.</li><li>Implements depth-first search algorithms in `DPOGenerator.py` for generating diverse dialogue outputs.</li><li>Utilizes JSONL format for efficient data storage and retrieval.</li></ul> |
| 📦 | **Dependencies**  | <ul><li>OpenAI api</li><li>Pydantic</li><li>Unidecode</li><li>PyPDF</li></ul> |

---

## 📁 Project Structure

```sh
└── /
    ├── core
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── components
    │   ├── loaders
    │   ├── logger.py
    │   └── processes
    ├── data
    │   ├── Pedagogy_docs
    │   ├── dialogues.json
    │   ├── dpo_dialogues.json
    │   ├── extracted_texts.json
    │   └── sample_data
    ├── example_usage.ipynb
    ├── prompts
    │   ├── choose_rule_prompt.txt
    │   ├── dialogue_prompt.txt
    │   ├── good_answer_and_question_prompt.txt
    │   └── rules.txt
    └── requirements.txt
```


### 📂 Project Index
<details open>
	<summary><b><code>/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='/example_usage.ipynb'>example_usage.ipynb</a></b></td>
				<td>- Demonstrates the process of extracting text from PDF documents and generating dialogues based on that text within the project<br>- It orchestrates the use of various components, such as text extraction and dialogue generation, to facilitate the creation of structured dialogue datasets<br>- This functionality is essential for the overall architecture, enabling the project to leverage extracted information for further analysis and dialogue modeling.</td>
			</tr>
			<tr>
				<td><b><a href='/requirements.txt'>requirements.txt</a></b></td>
				<td>- Lists of all python packages needed for the <code>datset_generation</code> part of the project.</td>
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
				<td>- Establishing a robust logging mechanism enhances the project's ability to monitor and troubleshoot application behavior<br>- By implementing a rotating file handler, it ensures that log files remain manageable in size while retaining a history of logs for analysis<br>- This functionality supports the overall architecture by providing critical insights into system performance and issues, thereby facilitating effective debugging and maintenance.</td>
			</tr>
			</table>
			<details>
				<summary><b>processes</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='/core/processes/DPOGenerator.py'>DPOGenerator.py</a></b></td>
						<td>- DPOGenerator facilitates the creation of Direct Preference Optimization training data by transforming dialogue datasets through pedagogical rules<br>- It systematically generates variations of dialogues, enhancing the dataset's richness for training purposes<br>- By applying a depth-first search approach, it ensures diverse dialogue outputs while managing the loading and saving of processed dialogues, ultimately contributing to the project's goal of improving dialogue systems.</td>
					</tr>
					<tr>
						<td><b><a href='/core/processes/DialogueGenerator.py'>DialogueGenerator.py</a></b></td>
						<td>- Facilitates the generation of educational dialogues between students and tutors by leveraging OpenAI's API<br>- It processes text documents, breaking them into manageable chunks, and constructs structured dialogues based on the content<br>- The resulting dialogues are saved in a JSONL format, enhancing the overall architecture by providing a dynamic interaction model for educational purposes.</td>
					</tr>
					<tr>
						<td><b><a href='/core/processes/ChunkExtractor.py'>ChunkExtractor.py</a></b></td>
						<td>- ChunkExtractor facilitates the extraction and processing of text from PDF files, transforming it into structured chunks for easier analysis and storage<br>- It preprocesses the text to enhance readability, removes unnecessary references, and preserves essential content like emails and URLs<br>- The results are saved in a JSONL format, ensuring efficient data management within the broader project architecture focused on document handling and analysis.</td>
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
						<td>- Dialogue serves as a foundational component for managing conversational interactions within the project<br>- It encapsulates multiple turns of dialogue, enabling the creation, serialization, and organization of conversations identified by unique IDs<br>- By facilitating the extraction and construction of chunk identifiers, it enhances the overall architecture's ability to handle complex dialogue structures, ensuring efficient data management and retrieval throughout the codebase.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/Chunk.py'>Chunk.py</a></b></td>
						<td>- Chunk serves as a fundamental component within the project, managing text segments identified by unique identifiers<br>- It facilitates the creation, serialization, and representation of these text chunks, ensuring efficient handling of document data<br>- By providing methods for ID extraction and JSON serialization, it enhances the overall architecture's capability to manipulate and store text data effectively, contributing to the project's functionality in processing and managing textual information.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/PedagogicalRules.py'>PedagogicalRules.py</a></b></td>
						<td>- Facilitates the management of pedagogical rules by providing a structured way to load, access, and iterate over rules defined in a text file<br>- It establishes a bidirectional mapping between rule indices and their corresponding texts, enabling users to retrieve rules by either index or text<br>- This functionality enhances the overall architecture by promoting organized rule handling within the educational framework of the project.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/BaseSubComponent.py'>BaseSubComponent.py</a></b></td>
						<td>- Defines a foundational class for subcomponents within the project, enabling dynamic attribute assignment through keyword arguments<br>- It establishes a framework for serialization and deserialization methods, which are essential for converting component data to and from JSON format<br>- This structure supports the overall architecture by promoting modularity and facilitating data interchange across various components in the codebase.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/Document.py'>Document.py</a></b></td>
						<td>- Document serves as a representation of a structured document composed of multiple text chunks, facilitating the management of document data, including identification and file information<br>- It enables the retrieval of specific chunks by their IDs and supports serialization to and from JSON format, enhancing data interchangeability<br>- This class plays a crucial role in the overall architecture by organizing and manipulating document content efficiently.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/DPODialogue.py'>DPODialogue.py</a></b></td>
						<td>- DPODialogue serves as a key component in managing dialogue interactions within the Direct Preference Optimization framework<br>- It facilitates the tracking of dialogue history, ID management, and the serialization and deserialization of dialogue data<br>- By integrating with other components, it ensures seamless data flow and interaction history, enhancing the overall functionality and user experience of the dialogue system.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/BaseComponent.py'>BaseComponent.py</a></b></td>
						<td>- Establishes a foundational class for components within the dataset generation pipeline, enabling consistent serialization to JSON and storage in JSONL format<br>- It defines essential methods for converting to and from JSON strings, alongside a mechanism for saving component data to a specified output file<br>- This structure promotes modularity and reusability across various components in the codebase.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/DPOTurn.py'>DPOTurn.py</a></b></td>
						<td>- DPOTurn serves as a representation of a conversational exchange in a Direct Preference Optimization context, encapsulating a student's question, a preferred answer, a less preferred answer, and the rule applied<br>- It facilitates the conversion of this data to and from JSON format, enabling seamless integration and communication within the broader architecture of the project, which likely focuses on enhancing conversational AI interactions.</td>
					</tr>
					<tr>
						<td><b><a href='/core/components/Turn.py'>Turn.py</a></b></td>
						<td>- Turn serves as a crucial component in managing conversational interactions between users and assistants<br>- It encapsulates the details of a conversation turn, including user messages and assistant responses, while providing functionality for serialization and deserialization to and from JSON format<br>- This enhances the overall architecture by enabling seamless storage and retrieval of conversation data, thereby facilitating effective communication within the application.</td>
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
						<td>- DPODialogueLoader serves as a crucial component in the project architecture, facilitating the loading and management of dialogue data from JSONL files<br>- It transforms raw dialogue entries into structured DPODialogue objects, maintains an index for efficient retrieval, and provides methods to extract unique dialogue IDs and their associated turns<br>- This enhances the overall functionality and accessibility of dialogue data within the system.</td>
					</tr>
					<tr>
						<td><b><a href='/core/loaders/BaseLoader.py'>BaseLoader.py</a></b></td>
						<td>- BaseLoader serves as a foundational component for dataset loaders within the project, specifically designed to handle JSONL files<br>- It establishes essential behaviors for managing datasets, including data retrieval and indexing, while enforcing a structure for subclasses to implement specific loading mechanisms<br>- This abstraction facilitates consistent data handling across various dataset implementations, enhancing the overall architecture's modularity and maintainability.</td>
					</tr>
					<tr>
						<td><b><a href='/core/loaders/DocumentLoader.py'>DocumentLoader.py</a></b></td>
						<td>- DocumentLoader serves as a crucial component in the project architecture, facilitating the loading and processing of data from JSONL files into structured Document objects<br>- It enables efficient retrieval of documents by their unique identifiers and maintains an index for quick access<br>- This functionality enhances data management and accessibility within the broader codebase, supporting various operations that rely on document data.</td>
					</tr>
					<tr>
						<td><b><a href='/core/loaders/DialogueLoader.py'>DialogueLoader.py</a></b></td>
						<td>- DialogueLoader serves as a crucial component in the project architecture by facilitating the loading and management of dialogue data from JSONL files<br>- It creates an index mapping dialogue IDs to their positions, enabling efficient retrieval of dialogues based on unique identifiers or associated document IDs<br>- This functionality enhances the overall data handling capabilities within the application, ensuring seamless access to dialogue information for further processing and analysis.</td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<details> <!-- prompts Submodule -->
		<summary><b>prompts</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='/prompts/choose_rule_prompt.txt'>choose_rule_prompt.txt</a></b></td>
				<td>- Facilitates the evaluation of pedagogical rules within a dialogue context by analyzing conversations between students and tutors<br>- It assesses the relevance of a specific pedagogical rule to an upcoming question-answer pair, providing a rating that reflects its appropriateness based on the conversation's progression<br>- This functionality enhances the overall architecture by ensuring that educational interactions are tailored and effective, promoting better learning outcomes.</td>
			</tr>
			<tr>
				<td><b><a href='/prompts/good_answer_and_question_prompt.txt'>good_answer_and_question_prompt.txt</a></b></td>
				<td>- Facilitates the adaptation of student-tutor interactions focused on assessment<br>- By guiding the generation of tailored student questions and tutor responses, it ensures adherence to pedagogical principles<br>- This process enhances the educational dialogue, promoting effective communication and learning outcomes within the broader project aimed at improving tutoring methodologies and student engagement.</td>
			</tr>
			<tr>
				<td><b><a href='/prompts/rules.txt'>rules.txt</a></b></td>
				<td>- Facilitating effective student engagement and personalized learning, the content outlines pedagogical strategies aimed at enhancing educational experiences<br>- By promoting self-reflection, adaptive teaching methods, and emotional support, it fosters a safe learning environment<br>- These principles are integral to the project's architecture, ensuring that educational interactions are tailored to individual needs, thereby optimizing student growth and understanding.</td>
			</tr>
			<tr>
				<td><b><a href='/prompts/dialogue_prompt.txt'>dialogue_prompt.txt</a></b></td>
				<td>- Facilitates the creation of an interactive dialogue between a student and a tutor focused on pedagogy<br>- By utilizing a reference text, it generates a structured conversation where the student initiates questions and the tutor provides detailed responses<br>- This approach enhances understanding of pedagogical concepts while ensuring engagement and clarity in communication, contributing to the overall educational objectives of the project.</td>
			</tr>
			</table>
		</blockquote>
	</details>
</details>

---
## 🚀 Getting Started

### ☑️ Prerequisites
Basically you should have installed Python 3.8 or higher.

Before getting started with , ensure your runtime environment meets the requirements in the `requirements.txt` file.
You can do so by running the following command:

```sh
❯ pip install -r requirements.txt
```

### 🤖 Usage
You can see an example on how to use the project in the `example_usage.ipynb` notebook.

---