# https://synthetichealth.github.io/synthea/
URLS_AND_DIRECTORIES = (
    (
        "https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_fhir_r4_sep2019.zip",
        "r4",
    ),
    (
        "https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_fhir_stu3_sep2019.zip",
        "stu3",
    ),
    (
        "https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_fhir_dstu2_sep2019.zip",
        "dstu2",
    ),
)


def main():
    for url, directory in URLS_AND_DIRECTORIES:
        wrangle(url, directory)


# download zip file at the given path
# unzip the .json files in the zip and put them in ./data/{directory}
# remove the zip file
def wrangle(url: str, directory: str) -> None:
    import urllib.request
    import zipfile
    import shutil
    import os

    # download the zip file
    print(f"Downloading {url}")
    zip_name = directory + ".zip"
    with urllib.request.urlopen(url) as response, open(zip_name, "wb") as out_file:
        out_file.write(response.read())

    # unzip the .json files
    print(f"Unzipping {directory}")
    with zipfile.ZipFile(zip_name, "r") as zip_ref:
        zip_ref.extractall(directory)

    # remove the zip file
    print(f"Removing {directory}")
    os.remove(zip_name)

    # walk ./{directory}/../ and move the json files to ./data/{directory}
    # i.e. ./stu3/fhir_stu3/*.json -> ./data/stu3/*.json
    print(f"Moving {directory}")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                os.rename(
                    os.path.join(root, file), os.path.join("data", directory, file)
                )

    # remove the directory
    print(f"Removing {directory}")
    shutil.rmtree(directory)


if __name__ == "__main__":
    main()