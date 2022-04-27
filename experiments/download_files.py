import requests
from tqdm import tqdm
import math
import zipfile
import os.path


def convert_size(size_bytes: int):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


def download(id: str, destination: str):
    if os.path.isfile(destination):
        print(f"File already downloaded in {destination}")
        print("Skipping download")
        return

    URL = "https://docs.google.com/uc?export=download"
    CHUNK_SIZE = 1024 * 1024  # 1024 * 1024 B = 1 MB

    session = requests.Session()

    response = session.get(URL, params={'id': id, "confirm": "t"}, stream=True)

    print(f"Start download in {destination}")
    bytes_downloaded = 0

    with open(destination, "wb") as f:
        for i, chunk in tqdm(enumerate(response.iter_content(CHUNK_SIZE))):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                if i % 100 == 0 and i != 0:
                    print(f"Downloaded {convert_size(bytes_downloaded)}")
                bytes_downloaded += CHUNK_SIZE


def extract(zip_file: str, dest: str):
    print(f"Start unzipping file in {dest}")
    with zipfile.ZipFile(zip_file, 'r') as zip_f:
        zip_f.extractall(dest)


def main():
    FILE_ID = "1fjk1qtS8G6aMP9xxPfBSCo9LvZB5xFh0"
    DL_DESTINATION = 'best_models_dl.zip'
    UNZIP_DESTINATION = "."
    download(FILE_ID, DL_DESTINATION)
    extract(DL_DESTINATION, UNZIP_DESTINATION)

    FILE_ID = "1JUeAXrWdPmtBPn6ulU9mp5oDFGcZH32G"
    DL_DESTINATION = 'results_dl.zip'
    UNZIP_DESTINATION = "."
    download(FILE_ID, DL_DESTINATION)
    extract(DL_DESTINATION, UNZIP_DESTINATION)


if __name__ == '__main__':
    main()
