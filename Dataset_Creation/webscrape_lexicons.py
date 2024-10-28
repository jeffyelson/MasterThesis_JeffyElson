from bs4 import BeautifulSoup
import time
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver import ChromeOptions
import sys

class MedicalLexiconWebScraper:
    def __init__(self, save_path, links):
        """
        This class is used to webscrape medical lexicons from the Harvard medical website to extract claims relevant to the lexicon
        @param save_path: Path to store the final list of medical lexicons in .txt format
        @param links: Final links used to extract the lexicons
        """
        self.save_path = save_path
        self.links = links
        self.final_links = self.generate_links()

    def generate_links(self):
        """
        This function is used to generate the final links for each alphabet from the Harvard Medical website
        @return: Final links for each alphabet
        """
        final_links = []
        for link, letter_range in zip(self.links, ['A-C', 'D-I', 'J-P', 'Q-Z']):
            start_letter, end_letter = letter_range.split('-')

            for letter in range(ord(start_letter), ord(end_letter) + 1):
                term = f"#{chr(letter)}-terms"
                final_links.append(link + term)

        return final_links

    def webscrape_lexicons(self):
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
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-web-security")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36")
        options.add_experimental_option("prefs", chrome_prefs)
        driver = webdriver.Chrome(options=options, service=self.get_chrome_service())
        health_terms = []
        for link in self.final_links:
            driver.get(link)
            time.sleep(1)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            time.sleep(1)
            for health in soup.find_all('p'):
                if health.find('strong'):
                    health_terms.append(health.find('strong').text.replace(":", "").rstrip())
        with open(self.save_path + "/" + "medical_lexicons" + '.txt', 'w+', encoding='utf-8') as file:
            for term in health_terms:
                file.write(term + "\n")
            file.close()

    def get_chrome_service(self):
        return Service(r"D:/chromedriver-win64/chromedriver.exe")


if __name__ == '__main__':
    save_path = "/"
    links = ["https://www.health.harvard.edu/a-through-c",
             "https://www.health.harvard.edu/d-through-i",
             "https://www.health.harvard.edu/j-through-p",
             "https://www.health.harvard.edu/q-through-z"]

    web_scraper = MedicalLexiconWebScraper(save_path, links)
    web_scraper.webscrape_lexicons()
