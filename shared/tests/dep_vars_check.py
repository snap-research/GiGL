import pathlib
import subprocess

DEP_VARS_FILE_PATH = pathlib.Path(__file__).parent.parent.parent / "dep_vars.env"


if __name__ == "__main__":
        with open(file=DEP_VARS_FILE_PATH, mode="r") as f:
            # Ensure we only have comments, empty lines, or lines with variable definitions
            for line in f.readlines():
                if line.startswith("#") or not line.strip(): # Is line a comment or empty?
                    continue
                if "=" not in line or ":=" in line: # := dictates runtime evaluation of the variable; = is static
                    raise ValueError(f"Invalid line found in `dep_vars.env`: {line}. Expected format: var=value")

            # Finally also check if the file is bash acceptable i.e. it can be sourced, so scripts like `install_scala_deps.sh`
            # can also use it
            cmd = f"source {DEP_VARS_FILE_PATH}"
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                raise ValueError(f"Error sourcing `dep_vars.env` file: {e}")

