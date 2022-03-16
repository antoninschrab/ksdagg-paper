"""
Download data in the directory data/ from 
https://drive.google.com/file/d/1CSj36IEW9x5_CCbiNXqBhqVqofiaGE0D/view?usp=sharing
This data has been obtained by running the generate_data_rbm.py and generate_data_nf.py scripts.
"""

from pathlib import Path
import requests, zipfile
from tqdm import tqdm

if Path('data').exists():
    raise ValueError("Directory ./data already exists!")

print('Starting download')
url = 'https://drive.google.com/u/0/uc?id=1CSj36IEW9x5_CCbiNXqBhqVqofiaGE0D&export=download&confirm=t'
filename = "data.zip"
response = requests.get(url, stream=True)
# progress bar
total_size_in_bytes= int(response.headers.get('content-length', 0))
block_size = 1024
progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
with open(filename, 'wb') as file:
    for data in response.iter_content(block_size):
        progress_bar.update(len(data))
        file.write(data)
progress_bar.close()
if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
    raise ValueError("Error: download failed. Try again.")
print('Download completed')

print('Starting unzip')
with zipfile.ZipFile(filename) as zf:
     for member in tqdm(zf.infolist(), desc='Extracting '):
         try:
             zf.extract(member, './')
         except zipfile.error as e:
             pass
print('Unzip completed')

# delete .zip file
Path("./data.zip").unlink()

