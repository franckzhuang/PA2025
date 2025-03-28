from concurrent.futures import ThreadPoolExecutor
import requests

def download_image(url: str, file_path: str):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download image from {url}, status code: {response.status_code}")


def download_images(urls: list[str], website_name: str):
    with ThreadPoolExecutor() as executor:
        for i, url in enumerate(urls):
            file_path = f"output/{website_name}_image_{i}.jpg"
            executor.submit(download_image, url, file_path)


def html_to_file(html, file_path: str):
    with open(file_path, "w") as f:
        f.write(str(html))
