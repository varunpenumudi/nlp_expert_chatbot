import kagglehub

def download_vecdb():
    # Download latest version
    path = kagglehub.dataset_download("varunpenumudi/arxiv-nlp-papers-dataset")

    print("Path to dataset files:", path)