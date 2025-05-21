# work in progress!
import os

def save_csv_upload(csv_file, path):
    if not os.path.exists(path):
        os.makedirs(path)
    # Save the uploaded CSV to the user_upload folder
    csv_save_path = os.path.join(path, 'uploaded_file.csv')
    with open(csv_save_path, 'wb') as f:
        f.write(csv_file.getvalue())

