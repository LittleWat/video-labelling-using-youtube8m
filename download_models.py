import os

import requests


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == "__main__":
    pca_file_id = '0B33vEXpXOGfmVUM1ZVdwd3VKbTA'
    pca_destination = os.path.join('pca_model', 'pcae.npz')
    download_file_from_google_drive(pca_file_id, pca_destination)
    print ('Finished downloading PCA model')

    yt8m_file_id_1 = '0B33vEXpXOGfmRlFEWmtlbGh4UjQ'
    yt8m_destination_1 = os.path.join('yt8m_model', 'model.ckpt-2833.data-00000-of-00001')
    download_file_from_google_drive(yt8m_file_id_1, yt8m_destination_1)

    yt8m_file_id_2 = '0B33vEXpXOGfmWlh5Vkg2R1VSNVE'
    yt8m_destination_2 = os.path.join('yt8m_model', 'model.ckpt-2833.index')
    download_file_from_google_drive(yt8m_file_id_2, yt8m_destination_2)

    yt8m_file_id_3 = '0B33vEXpXOGfmdk9DaWNmS2ZFb1k'
    yt8m_destination_3 = os.path.join('yt8m_model', 'model.ckpt-2833.meta')
    download_file_from_google_drive(yt8m_file_id_3, yt8m_destination_3)
    print ('Finished downloading yt8m model')
