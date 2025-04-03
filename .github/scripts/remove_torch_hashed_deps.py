from pathlib import Path

def remove_torch_dependencies(file_path):
    """Removes 'torch==' dependencies and their associated hash lines from a requirements file."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = []
    start_skipping = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("torch==") and "\\" in stripped:
            start_skipping = True
        # If we encounter a line that starts with "torch==" and has a backslash, we start skipping
        # We skip all subsequent hashes and comments (which dictate the dep structure for torch)
        elif start_skipping and (stripped.startswith("--hash=sha256:") or stripped.startswith("#")):
            continue
        else:
            start_skipping = False
            cleaned_lines.append(line)

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)

if __name__ == "__main__":
    for file_path in Path("requirements").glob("*.txt"):
        remove_torch_dependencies(file_path)
        print(f"Processed {file_path}")
