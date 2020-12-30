from google.cloud import storage

def upload_with_storage(upload_dir: str, gcs_bucket: str) -> None:
    """Uploads the upload_dir to the gs_bucket. Note that this method could
    results in timeout issues and is better not to use storage library and
    only rely on gsutil for copying files."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(gcs_bucket)
    for dirpath, _, filenames in os.walk(upload_dir):
        for name in filenames:
            filename = os.path.join(dirpath, name)
            blob = storage.Blob(os.path.join(upload_dir, name), bucket)
            with open(filename, 'rb') as f:
                blob.upload_from_file(f, num_retries=10, timeout=10 * 60)

