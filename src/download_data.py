import kaggle
import time

api = kaggle.KaggleApi()
api.authenticate()

DATASET = "mohammedabdeldayem/the-fake-or-real-dataset"

real_files = []
fake_files = []
page_token = None

# kaggle only gives you like 20-100 files per page so gotta loop through
while len(real_files) < 350 or len(fake_files) < 350:
    result = api.dataset_list_files(DATASET, page_token=page_token)
    for f in result.files:
        if "for-2sec/for-2seconds/training/real/" in f.name and len(real_files) < 350:
            real_files.append(f.name)
        elif "for-2sec/for-2seconds/training/fake/" in f.name and len(fake_files) < 350:
            fake_files.append(f.name)
    page_token = result.nextPageToken
    time.sleep(1)  # kept getting rate limited without this lol
    if not page_token:
        break

print(f"Found {len(real_files)} real files and {len(fake_files)} fake files")

for f in real_files:
    api.dataset_download_file(DATASET, file_name=f, path="data/real")

for f in fake_files:
    api.dataset_download_file(DATASET, file_name=f, path="data/fake")

print("Done downloading")
