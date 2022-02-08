"""
Download data in the directory data/ from 
https://drive.google.com/file/d/1CSj36IEW9x5_CCbiNXqBhqVqofiaGE0D/view?usp=sharing
This data has been obtained by running the generate_data_rbm.py and generate_data_nf.py scripts.
"""

from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(
    file_id="1CSj36IEW9x5_CCbiNXqBhqVqofiaGE0D", dest_path="./data.zip", unzip=True
)
Path("./data.zip").unlink()
