import os


def change_extensions(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            base = os.path.splitext(filename)[0]
            new_name = f"{base}.jpg"
            os.rename(
                os.path.join(folder_path, filename), os.path.join(folder_path, new_name)
            )
            print(f"Renamed: {filename} -> {new_name}")


if __name__ == "__main__":
    folder_path = input("Entrez le chemin du dossier : ")
    change_extensions(folder_path)
