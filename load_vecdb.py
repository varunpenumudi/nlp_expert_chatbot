import kagglehub
import shutil
import os

def download_vecdb():
    # Download latest version
    if os.path.exists('vecdb_contents'):
        return
    
    print("Loading Vector Database... ")
    path = kagglehub.dataset_download(
        "varunpenumudi/arxiv-nlp-papers-dataset",
    )
    print("Loaded Vector Database")

    shutil.copytree(path, 'vecdb_contents')
    os.remove('vecdb_contents/filtered_data.feather')