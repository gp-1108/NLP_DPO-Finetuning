from pypdf import PdfReader
import re
import json
from unidecode import unidecode
import os
import tqdm

class TextExtractor:
    EMAIL_TOKEN = "<EMAIL_TOKEN>"
    URL_TOKEN = "<URL_TOKEN>"
    SMALL_WORDS_TOKEN = "<SMALL_WORDS_TOKEN>"
    PUNCTUATION_TOKEN = "<PUNCTUATION_TOKEN>"
    CHUNK_MIN_LENGTH = 1000
    CHUNK_MAX_LENGTH = 7000

    def __init__(self, pdfs_path: str, output_dir: str):
        self.pdf_files = TextExtractor._load_pdf_files(pdfs_path)
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def extract_texts(self):
        """
        Extracts text from all PDF files, processes it, and saves the chunks to the output directory.
        """
        for pdf_file in tqdm.tqdm(self.pdf_files, desc="Extracting text from PDFs"):
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            if text:
                self.save_text(text, os.path.join(self.output_dir, os.path.basename(pdf_file).replace(".pdf", ".json")))

    @staticmethod
    def _load_pdf_files(pdfs_path: str) -> list[str]:
        """
        Loads all PDF file paths from the given directory.
        """
        pdf_files = []
        for root, _, filenames in os.walk(pdfs_path):
            for filename in filenames:
                if filename.endswith(".pdf"):
                    pdf_files.append(os.path.join(root, filename))
        return pdf_files

    def _pre_process(self, text: str) -> list[str]:
        """
        Pre-processes the text by removing emails, URLs, small words, and punctuation,
        then re-adds them after processing.
        """
        # Removing unicode characters
        text = unidecode(text)

        # Remove references
        text = self._remove_references(text)

        # Removing interfering elements
        text, emails = self._remove_emails(text)
        text, urls = self._remove_urls(text)
        text, small_words = self._remove_small_words(text, 4)
        text, punctuation = self._remove_punctuation(text)

        # Re-adding the removed elements
        text = self._add_small_words(text, small_words)
        text = self._add_urls(text, urls)
        text = self._add_emails(text, emails)

        chunks = text.split(self.PUNCTUATION_TOKEN)

        # Adding back the punctuation that was removed
        for i in range(len(chunks)):
            if i < len(punctuation):
                chunks[i] = chunks[i] + punctuation[i]

        # Unifying chunks that are too small
        chunks = self._unify_chunks(chunks)
        chunks = self._polish_chunks(chunks)

        return chunks

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
    
    def _remove_references(self, text: str) -> str:
        """
        Removes references from the text.

        Args:
            text (str): The text to remove references from.

        Returns:
            str: The text without references.
        """
        # First we need to know how many times the "references" word appears in the text
        # if it is more than 10, it means that the text has multiple references.
        # as of right now this is not supported and the text will be returned as is.
        search_text = text.lower()
        references_count = search_text.count("references")
        if references_count > 10 or references_count == 0:
            return text
        
        # Now we need to find the index at which the "references word appears".
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
        
        return text[:references_index]
    

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

    def save_text(self, text: str, json_path: str):
        """
        Saves the processed text chunks into a JSON file.
        """
        chunks = self._pre_process(text)

        if not chunks or sum(len(chunk.split()) for chunk in chunks) < 200:
            print(f"Text for file {json_path} is probably corrupted or not useful.")
            return

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
