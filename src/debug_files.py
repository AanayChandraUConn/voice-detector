import kaggle

api = kaggle.KaggleApi()
api.authenticate()
DATASET = "mohammedabdeldayem/the-fake-or-real-dataset"

page_token = None
checked = 0
found_folders = set()

while checked < 2000:  # only check first 2000 files, should be plenty
    result = api.dataset_list_files(DATASET, page_token=page_token)
    for f in result.files:
        checked += 1
        parts = f.name.split('/')
        if len(parts) > 1:
            found_folders.add('/'.join(parts[:-1]))
    page_token = result.nextPageToken
    if not page_token:
        break

print(f"Checked {checked} files")
for folder in sorted(found_folders):
    print(folder)
