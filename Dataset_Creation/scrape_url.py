import pandas as pd
df = pd.read_excel('Claims_withGeminiAnnotation.xlsx')
# Path to the output text file
output_file_path = 'D:/Master Thesis/src/output_urls.txt'

with open(output_file_path, 'w') as file:
    # Iterate over each entry in the 'url' column
    for url_entry in df['url']:
        # Check if the URL entry is not empty or null   
        if pd.notnull(url_entry) and url_entry.strip():
            # Split URLs if '||' is present, remove leading and trailing spaces from each URL
            urls = [url.strip() for url in url_entry.split('||')]

            # Process each URL based on its starting substring
            for url in urls:
                # Remove trailing spaces before checking the URL prefix
                url = url.rstrip()
                if url.startswith('https://www.ncbi.nlm.nih.gov/'):
                    url += '?report=printable'
                elif url.startswith('https://pubmed.ncbi.nlm.nih.gov/'):
                    url += '?format=pubmed'

                # Write each processed URL to the file, each on a new line
                file.write(url + '\n')

print("URLs have been processed and written to:", output_file_path)