from pypdf import PdfReader
import re
from unidecode import unidecode
import os
import tqdm
from ..components import Document
from ..components import Chunk
from ..loaders import DocumentLoader

class ChunkExtractor:
    """
    A class for extracting and processing text from PDF files into structured chunks.
    This class handles the extraction of text from PDF files, processes the text into
    manageable chunks, and saves the results in a JSONL file format. It includes
    functionality for text preprocessing, reference removal, and handling of special
    content like emails and URLs.
    Attributes:
        EMAIL_TOKEN (str): Token used to temporarily replace email addresses during processing
        URL_TOKEN (str): Token used to temporarily replace URLs during processing
        SMALL_WORDS_TOKEN (str): Token used to temporarily replace small words during processing
        PUNCTUATION_TOKEN (str): Token used to temporarily replace punctuation during processing
        CHUNK_MIN_LENGTH (int): Minimum length threshold for text chunks (default: 1000)
        CHUNK_MAX_LENGTH (int): Maximum length threshold for text chunks (default: 7000)
        pdfs_path (str): Path to the directory containing PDF files to process
        output_jsonl (str): Path where the output JSONL file will be saved
        chunk_min_length (int, optional): Minimum length for text chunks. Defaults to 1000
        chunk_max_length (int, optional): Maximum length for text chunks. Defaults to 7000
    Example:
        >>> extractor = ChunkExtractor("path/to/pdfs", "output.jsonl")
        >>> extractor.extract_texts()
        - The class handles PDF processing recursively in the specified directory
        - Documents are processed only once (duplicates are skipped)
        - Text chunks are processed to maintain coherence and readability
        - Special content (emails, URLs) is preserved using token replacement
    """
    EMAIL_TOKEN = "<EMAIL_TOKEN>"
    URL_TOKEN = "<URL_TOKEN>"
    SMALL_WORDS_TOKEN = "<SMALL_WORDS_TOKEN>"
    PUNCTUATION_TOKEN = "<PUNCTUATION_TOKEN>"
    CHUNK_MIN_LENGTH = 1000 # Minimum length of a chunk
    CHUNK_MAX_LENGTH = 7000 # Maximum length of a chunk

    def __init__(self,
                 pdfs_path: str,
                 output_jsonl: str,
                 chunk_min_length: int = 1000,
                 chunk_max_length: int = 7000):
        self.pdf_files = ChunkExtractor._load_pdf_files(pdfs_path)
        self.output_jsonl = output_jsonl
        self.already_processed = DocumentLoader(output_jsonl)
        self.CHUNK_MAX_LENGTH = chunk_max_length
        self.CHUNK_MIN_LENGTH = chunk_min_length
        self.id_counter = 0

    def extract_texts(self):
        """
        Extracts text from all PDF files, processes it, and saves the documents to a JSONL file.
        """
        for pdf_file in tqdm.tqdm(self.pdf_files, desc="Extracting text from PDFs"):
            try:
                self.extract_single_text(pdf_file)
            except Exception as e:
                print(f"Error while processing file {pdf_file}: {e}")
    
    def extract_single_text(self, pdf_file: str):
        """
        Extracts text from a single PDF file and saves it as a Document instance in the database.

        This method reads the PDF file page by page, extracts the text content, and processes it
        into a Document object. If the text extraction and processing are successful, the document
        is saved to the database.

        Args:
            pdf_file (str): The file path to the PDF file to be processed.

        Returns:
            None

        Note:
            The method will only save the document if both text extraction and document processing
            are successful (i.e., if they return non-empty/non-None values).
        """
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        if text:
            document = self.process_text_to_document(text, pdf_file)
            if document:
                document.save()

    @staticmethod
    def _load_pdf_files(pdfs_path: str) -> list[str]:
        """
        Recursively search for PDF files in the specified directory and its subdirectories.

        Args:
            pdfs_path (str): The root directory path to search for PDF files.

        Returns:
            list[str]: A list of full paths to all PDF files found in the directory tree.
        """
        pdf_files = []
        for root, _, filenames in os.walk(pdfs_path):
            for filename in filenames:
                if filename.endswith(".pdf"):
                    pdf_files.append(os.path.join(root, filename))
        return pdf_files

    def process_text_to_document(self, text: str, pdf_file: str) -> Document:
        """
        Process a text and its corresponding PDF file into a Document object.

        This method performs the following operations:
        1. Generates a unique document ID
        2. Checks if the document has already been processed
        3. Pre-processes the text into chunks
        4. Creates a Document object with the processed chunks

        Args:
            text (str): The text content to be processed
            pdf_file (str): Path to the PDF file associated with the text

        Returns:
            Document: A Document object containing the processed chunks and metadata
                     Returns None if:
                     - Document was already processed
                     - Text is corrupted or too short (less than 200 words)

        Raises:
            ValueError: If a document with the same ID exists but with a different file name
        """
        doc_int_id = self._generate_id()
        doc_id = Document.get_id(doc_int_id)
        if doc_id in self.already_processed and \
            self.already_processed.get_document_by_id(doc_id).file_name != os.path.basename(pdf_file):
            raise ValueError(f"ERROR: Document with ID {doc_id} already processed with a different file name. \
                             Do not re-run the script with new data if previous data is already present.")
        if doc_id in self.already_processed:
            print(f"Document with ID {doc_id} already processed.")
            return None

        chunks = self._pre_process(text)
        if not chunks or sum(len(chunk.split()) for chunk in chunks) < 200:
            print(f"Text for file {pdf_file} is probably corrupted or not useful.")
            return None

        document_chunks = [
            Chunk(text=chunk, id=Chunk.get_id(doc_id, i)) for i,chunk in enumerate(chunks)
        ]
        document = Document(
            output_file=self.output_jsonl,
            file_name=os.path.basename(pdf_file),
            id=doc_id,
            chunks=document_chunks
        )
        return document

    def _generate_id(self) -> int:
        """
        Generates a unique ID for this run.
        Ids are generated sequentially starting from 1.
        """
        self.id_counter += 1
        return self.id_counter

    def _pre_process(self, text: str) -> list[str]:
        """
        Preprocesses the input text by applying a series of transformations and returns a list of text chunks.

        The preprocessing steps include:
        1. Converting text to ASCII using unidecode
        2. Removing and storing references
        3. Extracting and storing emails
        4. Extracting and storing URLs
        5. Removing and storing small words (less than 4 characters)
        6. Removing and storing punctuation
        7. Restoring small words
        8. Restoring URLs
        9. Restoring emails
        10. Splitting text into chunks based on punctuation
        11. Unifying and polishing the resulting chunks

        Args:
            text (str): The input text to be preprocessed

        Returns:
            list[str]: A list of preprocessed text chunks
        """
        text = unidecode(text)
        text = self._remove_references(text)
        text, emails = self._remove_emails(text)
        text, urls = self._remove_urls(text)
        text, small_words = self._remove_small_words(text, 4)
        text, punctuation = self._remove_punctuation(text)
        text = self._add_small_words(text, small_words)
        text = self._add_urls(text, urls)
        text = self._add_emails(text, emails)

        chunks = text.split(self.PUNCTUATION_TOKEN)
        for i in range(len(chunks)):
            if i < len(punctuation):
                chunks[i] = chunks[i] + punctuation[i]

        chunks = self._unify_chunks(chunks)
        chunks = self._polish_chunks(chunks)

        return chunks
        
    def _remove_references(self, text: str) -> str:
        """
        Remove references section from the text.
        This method attempts to identify and remove the references section from a given text by
        locating the word "references" that typically indicates the beginning of a bibliography
        or reference section in academic papers.
        The method performs a case-insensitive search and handles different variations of how
        "references" might appear in the text (with spaces before/after, or as part of other words).
        It removes everything from the identified "references" marker to the end of the text.

        More specifically, the method looks for the last occurrence of the word "references" in the
        text and removes everything after that point. If the word "references" is part of another
        word (e.g., "coreferences"), it will not be considered as a valid reference section marker.
        
        Args:
            text (str): The input text from which to remove the references section.
        Returns:
            str: The text with the references section removed. If no valid references section
                 is found, returns the original text unchanged.
        Notes:
            - The method checks for false positives where "references" might be part of other
              words like "preferences" or "coreferences".
            - It always takes the last occurrence of "references" in the text, as reference
              sections typically appear at the end.
            - If multiple occurrences of "references" exist, only the content after the last
              valid occurrence is removed.
        """
        # First we need to know how many times the "references" word appears in the text
        search_text = text.lower()
        references_count = search_text.count("references")
        if references_count == 0:
            return text
        
        # Now we need to find the index at which the "references" word appears.
        # To do it we will progressively look at " references " " references" "references"
        # whichever matches first will be the index at which the references word appears.
        # we will always take the last match as the references are usually at the end of the text.
        references_index = -1
        if " references " in search_text:
            references_index = search_text.rindex(" references ")
        elif " references" in search_text:
            references_index = search_text.rindex(" references")
        elif "references" in search_text:
            references_index = search_text.rindex("references")
            # In this case we need to check if the word "references" is not part of another word.
            words = [
                "coreferences",
                "crossreferences",
                "dereferences",
                "georeferences",
                "preferences",
                "references",
                "subreferences"
            ]
            max_length = max(len(word) for word in words)
            # Now let's see what the program matched
            left_idx = references_index - max_length if references_index - max_length > 0 else 0
            right_idx = references_index + len("references")
            matched_word = search_text[left_idx:right_idx]
            for word in words:
                if word in matched_word:
                    references_index = -1
                    break
        return text[:references_index]

    def _remove_emails(self, text: str):
        """
        Replaces emails with a token and stores them for later re-insertion.
        """
        email_regex = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
        emails = re.findall(email_regex, text)
        text = re.sub(email_regex, self.EMAIL_TOKEN, text)
        return text, emails

    def _remove_urls(self, text: str):
        """
        Replaces URLs with a token and stores them for later re-insertion.
        """
        url_regex = r"https?://[^\s]+"
        urls = re.findall(url_regex, text)
        text = re.sub(url_regex, self.URL_TOKEN, text)
        return text, urls

    def _remove_punctuation(self, text: str):
        """
        Removes punctuation from the text.
        """
        punctuation = re.findall(r"[.!?;]", text)
        text = re.sub(r"[.!?;]", self.PUNCTUATION_TOKEN, text)
        return text, punctuation

    def _remove_small_words(self, text: str, max_length: int):
        """
        Removes small words that could interfere with chunking.
        """
        small_word_regex = r"\b\w{1," + str(max_length) + r"}\.(?!\n)"
        small_words = re.findall(small_word_regex, text)
        text = re.sub(small_word_regex, self.SMALL_WORDS_TOKEN, text)
        return text, small_words

    def _add_emails(self, text: str, emails: list):
        """
        Replaces email tokens with the original emails.
        """
        for email in emails:
            text = text.replace(self.EMAIL_TOKEN, email, 1)
        return text

    def _add_urls(self, text: str, urls: list):
        """
        Replaces URL tokens with the original URLs.
        """
        for url in urls:
            text = text.replace(self.URL_TOKEN, url, 1)
        return text

    def _add_small_words(self, text: str, small_words: list):
        """
        Replaces small word tokens with the original small words.
        """
        for word in small_words:
            text = text.replace(self.SMALL_WORDS_TOKEN, word, 1)
        return text

    def _unify_chunks(self, chunks: list[str]) -> list[str]:
        """
        Unifies chunks of text based on certain conditions to create more coherent text segments.

        This method combines consecutive chunks if they meet specific criteria:
        - The last unified chunk is shorter than CHUNK_MIN_LENGTH, or
        - The current chunk starts with newline/tab/space characters, or
        - The last chunk ends with newline/tab/space characters
        AND the combined length is less than CHUNK_MAX_LENGTH

        Args:
            chunks (list[str]): List of text chunks to be unified

        Returns:
            list[str]: List of unified text chunks where appropriate chunks have been combined
        """
        if len(chunks) <= 1:
            return chunks

        unified_chunks = [chunks[0]]
        matches = ["\n", "\t", " "]

        for i in range(1, len(chunks)):
            last_chunk = unified_chunks[-1]
            current_chunk = chunks[i]
            if (len(last_chunk) < self.CHUNK_MIN_LENGTH or
                current_chunk.startswith(tuple(matches)) or
                last_chunk.endswith(tuple(matches))) and \
                len(last_chunk) + len(current_chunk) < self.CHUNK_MAX_LENGTH:
                unified_chunks[-1] += ' ' + current_chunk
            else:
                unified_chunks.append(current_chunk)
        return unified_chunks

    def _polish_chunks(self, chunks: list[str]) -> list[str]:
        """
        This function will polish the chunks by removing not useful chunks based on some criteria.
        It will return a list of polished chunks.

        Args:
            chunks (list[str]): A list of text chunks.

        Returns:
            list[str]: A list of polished text chunks.
        """
        polished_chunks = []
        for chunk in chunks:
            chunk = chunk.replace("\n", "").strip()

            # If the content of alphanumerical characters is less than 50% of the chunk
            # then we will discard the chunk.
            if not chunk or len(re.sub(r"[^a-zA-Z0-9]", "", chunk)) / len(chunk) < 0.5:
                continue
            polished_chunks.append(chunk)
        return polished_chunks