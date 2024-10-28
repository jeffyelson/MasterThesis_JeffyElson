import re

from bs4 import BeautifulSoup
import time

from selenium.common import NoSuchElementException
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from get_chrome_driver import GetChromeDriver
import pandas as pd
from selenium.webdriver.common.by import By

get_driver = GetChromeDriver()
get_driver.install()

class MedicalCorpusWebScraper:


    def __init__(self, save_path):
        """
        This class is used to webscrape medical lexicons from the Harvard medical website to extract claims relevant to the lexicon
        @param save_path: Path to store the final list of medical lexicons in .txt format
        @param links: Final links used to extract the lexicons
        """
        self.save_path = save_path
    def webscrape_corpus(self):
        """
        This function is used to webscrape medical lexicons from the Harvard medical website to extract claims relevant to the lexicon
        @return: Returns nothing
        """
        options = ChromeOptions()
        chrome_prefs = {
            "download.prompt_for_download": False,
            "plugins.always_open_pdf_externally": True,
            "download.open_pdf_in_system_reader": False,
            "profile.default_content_settings.popups": 0,
            "download.default_directory": self.save_path
        }

        options.add_argument('--headless')
        # options.add_argument("--start-maximized")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-web-security")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36")
        options.add_experimental_option("prefs", chrome_prefs)
        driver = webdriver.Chrome(options=options)
        # Path to the text file containing URLs
        input_file_path = 'D:/Master Thesis/src/output_urls.txt'

        urls = []

        with open(input_file_path, 'r') as file:
            for line in file:
                stripped_line = line.strip()
                if stripped_line:
                    urls.append(stripped_line)
        urls = urls[99:]
        for num,link in enumerate(urls):
            try:
                num+=99
                print(num,link)
                driver.get(link)
                time.sleep(1)
                print("Scraping URL:", link)
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                time.sleep(1)
                soup = soup.find('body')
                text = soup.get_text(separator='\n', strip=True)
                lines = text.split('\n')

                long_lines = [line for line in lines if len(line) >= 25]

                # Join the remaining lines back into a single string with new lines
                filtered_text = '\n'.join(long_lines)

                if link.startswith('https://pubmed.ncbi.nlm.nih.gov/'):
                    text_cleaned = clean_text(filtered_text)
                    final_text = extract_abstract(text)
                    print('Abstract:', final_text)
                else:
                    final_text = clean_text(filtered_text)
                    print(final_text)
                print('----------------------')
                with open(save_path+'doc'+ str(num) + '.txt', 'w+', encoding='utf-8') as file:
                    file.writelines(final_text)
                    file.close()

            except Exception as e:
                print(e)
                continue

def clean_text(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    cookies_pattern = re.compile(r'\bcookies\b', re.IGNORECASE)
    lines = text.split('\n')

    cleaned_lines = []
    for line in lines:
        if url_pattern.search(line):
            continue
        if cookies_pattern.search(line):
            continue
        cleaned_lines.append(line)

    cleaned_text = '\n'.join(cleaned_lines)

    return cleaned_text


import re

def extract_abstract(text):

    abstract_pattern = re.compile(r'AB  - (.*?)\n(?=[A-Z]+\s+- )', re.DOTALL)
    match = abstract_pattern.search(text)
    if match:
        abstract_text = match.group(1)
        abstract_text = re.sub(r'\n\s+', ' ', abstract_text)
        return abstract_text.strip()
    else:
        return "Abstract not found or does not follow the expected format."






if __name__ == '__main__':
    save_path = "D:/Master Thesis/src/corpus/"
    web_scraper = MedicalCorpusWebScraper(save_path)
    web_scraper.webscrape_corpus()