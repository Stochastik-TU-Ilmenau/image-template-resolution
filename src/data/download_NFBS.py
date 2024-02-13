"""
Download and extract the "NFBS Skull-Stripped Repository" from:
    http://preprocessed-connectomes-project.org/NFB_skullstripped

This contains 3d mri images of human heads and their skull stripped versions.
Creates folder "data/raw" if it does not yet exist!
"""

import os
import requests
from tqdm import tqdm
import tarfile

data_path = 'data/raw/' # relative to Makefile
download_url = 'https://fcp-indi.s3.amazonaws.com/data/Projects/RocklandSample/NFBS_Dataset.tar.gz'

# create download folder
if not os.path.exists(data_path):
    os.makedirs(data_path)

# download NFBS_Dataset.tar.gz
response = requests.get(download_url, stream=True)
total_size_in_bytes= int(response.headers.get('content-length', 0))
progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
with open(data_path + 'NFBS_Dataset.tar.gz', 'wb') as file:
    for data in response.iter_content(chunk_size=1024):
        progress_bar.update(len(data))
        file.write(data)
progress_bar.close()

# extract NFBS_Dataset.tar.gz
nfbs_tar_gz = tarfile.open(data_path + 'NFBS_Dataset.tar.gz')
nfbs_tar_gz.extractall(data_path)

# remove NFBS_Dataset.tar.gz:
os.remove(data_path + 'NFBS_Dataset.tar.gz')
