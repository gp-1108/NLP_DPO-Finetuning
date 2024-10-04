from pypdf import PdfReader
import re
import json
from unidecode import unidecode

class TextExtractor:
    EMAIL_TOKEN = "<EMAIL_TOKEN>"
    URL_TOKEN = "<URL_TOKEN>"
    SMALL_WORDS_TOKEN = "<SMALL_WORDS_TOKEN>"
    PUNCTUATION_TOKEN = "<PUNCTUATION_TOKEN>"
    CHUNK_MIN_LENGTH = 1000
    CHUNK_MAX_LENGTH = 7000
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        file = PdfReader(pdf_path)
        self.text = ""
        for page in file.pages:
            self.text += page.extract_text()
    
    def _pre_process(self):
        # Removing unicode characters
        self.text = unidecode(self.text)
        # Removing interfering punctuation
        self._remove_emails()
        self._remove_urls()
        self._remove_small_words(3)

        # Preparing the text for chunking
        self._remove_punctuation()
        
        # Re-adding the removed punctuation
        self._add_small_words()
        self._add_urls()
        self._add_emails()
        
        chunks = self.text.split(self.PUNCTUATION_TOKEN)
        chunks = [chunk for chunk in chunks if chunk]

        # Adding back the punctuation that was removed
        for i in range(len(chunks)):
            chunks[i] = chunks[i] + (self.punctuation.pop(0) if self.punctuation else "")
        
        # Unifying chunks that are too small
        chunks = self._unify_chunks(chunks)

        chunks = self._polish_chunks(chunks)

        self.chunks = chunks
        return chunks 
    
    def _polish_chunks(self, chunks: list[str]) -> list[str]:
        """
        This function will polish the chunks by removing any \n in the phrases
        and removing any leading or trailing spaces (or unicode characters).
        """
        for i in range(len(chunks)):
            chunks[i] = chunks[i].replace("\n", " ")
            chunks[i] = chunks[i].strip()
        return chunks

    
    def get_chunks(self):
        """
        This function will return the chunks of text.
        """
        return self.chunks
    
    def _unify_chunks(self, chunks: list[str]) -> list[str]:
        """
        This function will merge subsequent chunks that are too small or if the chunks ends/starts
        with \n or space.

        Args:
            chunks (list[str]): A list of text chunks.
        
        Returns:
            list[str]: A list of text chunks with some of them merged.
        """
        if len(chunks) <= 1:
            return chunks
            
        unified_chunks = [chunks[0]]
        matches = ["\n", "\t"]

        for i in range(1, len(chunks)):
            if (len(unified_chunks[-1]) < self.CHUNK_MIN_LENGTH \
                or chunks[i][0] in matches \
                or chunks[i-1][-1] in matches) \
                and len(unified_chunks[-1]) + len(chunks[i]) < self.CHUNK_MAX_LENGTH:
                unified_chunks[-1] += chunks[i]
            else:
                unified_chunks.append(chunks[i])
        return unified_chunks
    
    def _remove_punctuation(self):
        """
        This function will remove all punctuation from the text.
        """
        # Define punctuation to be removed: . , ! ? ;
        self.punctuation = re.findall(r"[.!?;]", self.text)
        self.text = re.sub(r"[.!?;]", self.PUNCTUATION_TOKEN, self.text)
    
    def _add_punctuation(self):
        """
        This function will add the punctuation that was removed.
        """
        while self.PUNCTUATION_TOKEN in self.text and self.punctuation:
            self.text = self.text.replace(self.PUNCTUATION_TOKEN, self.punctuation.pop(0), 1)

    def _remove_small_words(self, max_length: int):
        """
        This function will remove all small words that end with a period or are part of 
        abbreviations like 'et al.' or 'e.g.'.
        """
        # Match small words followed by period and any other characters (like space or more text)
        small_word_regex = r"\b\w{0," + str(max_length) + r"}\.(?!\n)(?:\s|[^a-zA-Z\n]|\b\w{1}\.)*"
        self.small_words = re.findall(small_word_regex, self.text)
        self.text = re.sub(small_word_regex, self.SMALL_WORDS_TOKEN, self.text)

    
    def _add_small_words(self):
        """
        This function will add the small words that were removed.
        """
        while self.SMALL_WORDS_TOKEN in self.text and self.small_words:
            self.text = self.text.replace(self.SMALL_WORDS_TOKEN, self.small_words.pop(0), 1)
        
    def _remove_emails(self):
        """
        This function will replace all emails with the special token EMAIL and
        save the emails in a list to later replace the token with the email once
        the chunks are divided by the stop words.
        """ 
        email_regex = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}"
        self.emails = re.findall(email_regex, self.text)
        self.text = re.sub(email_regex, self.EMAIL_TOKEN, self.text)
    
    def _add_emails(self):
        """
        This function will replace the EMAIL token with the email.
        """
        while self.EMAIL_TOKEN in self.text and self.emails:
            self.text = self.text.replace(self.EMAIL_TOKEN, self.emails.pop(0), 1)
    
    def _remove_urls(self): 
        """
        This function will replace all URLs with the special token URL and
        save the URLs in a list to later replace the token with the URL once
        the chunks are divided by the stop words.
        """
        url_regex = r"https?://(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\\s]*)?"
        self.urls = re.findall(url_regex, self.text)
        self.text = re.sub(url_regex, self.URL_TOKEN, self.text)
    
    def _add_urls(self):
        """
        This function will replace the URL token with the URL.
        """
        while self.URL_TOKEN in self.text and self.urls:
            self.text = self.text.replace(self.URL_TOKEN, self.urls.pop(0), 1)
    
    def save_text(self, json_path: str):
        """
        This function will save the text in a json file.
        """
        chunks = self._pre_process()
        with open(json_path, 'w') as f:
            json.dump(chunks, f, indent=2)
