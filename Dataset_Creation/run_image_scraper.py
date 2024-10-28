#Import libraries
import os
import concurrent.futures
from image_scraper import GoogleImageScraper


def worker_thread(search_key):
    image_scraper = GoogleImageScraper(
        webdriver_path,
        image_path,
        search_key,
        number_of_images,
        headless,
        min_resolution,
        max_resolution,
        max_missed)
    image_urls = image_scraper.find_image_urls()
    image_scraper.save_images(image_urls, keep_filenames)

    #Release resources
    del image_scraper

def remove_empty_folders(folder_path):
    if not os.path.isdir(folder_path):
        return "Invalid path"

    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    for subfolder in subfolders:
        files = [f.name for f in os.scandir(subfolder) if f.is_file()]
        if not files:
            os.rmdir(subfolder)

if __name__ == "__main__":
    #Define file path
    webdriver_path = os.path.normpath(os.path.join(os.getcwd()))
    image_path = os.path.normpath(os.path.join(os.getcwd(), 'images'))

    #Read webscraped lexicon file

    medical_lexicon = []
    with open("medical_lexicons.txt", "r") as file:
        for line in file:
            med = "myths on "+ line.strip()
            medical_lexicon.append(med)

    search_keys = list(set(medical_lexicon))

    #Parameters
    number_of_images = 5                # Desired number of images
    headless = True                     # True = No Chrome GUI
    min_resolution = (0, 0)             # Minimum desired image resolution
    max_resolution = (9999, 9999)       # Maximum desired image resolution
    max_missed = 10                     # Max number of failed images before exit
    number_of_workers = 1               # Number of "workers" used
    keep_filenames = False              # Keep original URL image filenames

    #Run each search_key in a separate thread
    #Automatically waits for all threads to finish
    #Removes duplicate strings from search_keys
    with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_workers) as executor:
        executor.map(worker_thread, search_keys)

    #Remove empty folders
    path = "D:/Master Thesis/src/images/"
    msg = remove_empty_folders(path)
    print(msg)