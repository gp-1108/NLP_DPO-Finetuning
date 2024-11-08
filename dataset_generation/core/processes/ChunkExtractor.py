from pypdf import PdfReader
import re
from unidecode import unidecode
import os
import tqdm
from ..components import Document
from ..components import Chunk

class ChunkExtractor():
    EMAIL_TOKEN = "<EMAIL_TOKEN>"
    URL_TOKEN = "<URL_TOKEN>"
    SMALL_WORDS_TOKEN = "<SMALL_WORDS_TOKEN>"
    PUNCTUATION_TOKEN = "<PUNCTUATION_TOKEN>"
    CHUNK_MIN_LENGTH = 1000
    CHUNK_MAX_LENGTH = 7000

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
        pdf_files = []
        for root, _, filenames in os.walk(pdfs_path):
            for filename in filenames:
                if filename.endswith(".pdf"):
                    pdf_files.append(os.path.join(root, filename))
        return pdf_files

    def process_text_to_document(self, text: str, pdf_file: str) -> Document:
        """
        Processes the extracted text and creates a Document instance.
        """
        doc_int_id = self._generate_id()
        doc_id = Document.get_id(doc_int_id)
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
        """
        self.id_counter += 1
        return self.id_counter

    def _pre_process(self, text: str) -> list[str]:
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
        Removes references from the text.

        Args:
            text (str): The text to remove references from.

        Returns:
            str: The text without references.
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
        Merges subsequent chunks that are too small or improperly split.
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