FILES_TO_CHECK = [
    "requirements/darwin_arm64_requirements_unified.txt",
    "requirements/dev_darwin_arm64_requirements_unified.txt",
    "requirements/dev_linux_cpu_requirements_unified.txt",
    "requirements/dev_linux_cuda_requirements_unified.txt",
    "requirements/linux_cpu_requirements_unified.txt",
    "requirements/linux_cuda_requirements_unified.txt",
]

if __name__ == "__main__":
    for file in FILES_TO_CHECK:
        with open(file=file, mode="r") as f:
            contents = f.read()
            if "torch==" in contents:
                raise ValueError(
                    f"""
                Found issue in file: {file}
                Ensure that requirements files do not contain `torch` as a listed dependency. This may cause various issues.
                Some packages like PyG require a prior installation of torch to install, thus having it listed as a dep
                might result in issues - even seg faults if the version installed != one that pyg or dgl expect. Ensure
                compatible versions of pytorch are pre-installed in `install_py_deps.sh`.
                If generating a hashed req and torch is automatically added, delete it manually.
                """
                )
