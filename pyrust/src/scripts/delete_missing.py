import os
import argparse


def get_filenames(directory):
    try:
        return set(os.listdir(directory))
    except FileNotFoundError:
        print(f"Folder '{directory}' doesn't exist.")
        exit(1)


def delete_missing(source_dir, target_dir, dry_run=True):
    source_files = get_filenames(source_dir)
    target_files = get_filenames(target_dir)

    to_delete = [f for f in target_files if f not in source_files]

    if not to_delete:
        print("No files to delete.")
        return

    print(f"Files to delete in '{target_dir}':")
    for f in to_delete:
        print(f"  - {f}")

    if dry_run:
        print("\nMode dry-run: no files deleted.")
        return

    for f in to_delete:
        path = os.path.join(target_dir, f)
        try:
            os.remove(path)
            print(f"Deleted : {path}")
        except Exception as e:
            print(f"Error while deleting {path} : {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deletes files in the target folder that are not in the source folder."
    )
    parser.add_argument("source_dir", help="Path to folder 1 (reference source)")
    parser.add_argument("target_dir", help="Path to folder 2 (target for deletion)")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Activate to performs the deletion. Without this option, the script performs a dry-run..",
    )

    args = parser.parse_args()
    delete_missing(args.source_dir, args.target_dir, dry_run=not args.execute)
