import subprocess
from typing import List

from pydantic import BaseModel, HttpUrl, TypeAdapter

from lib.misc import datasets_root_path


def to_url(str_url: str) -> HttpUrl:
    adapter = TypeAdapter(HttpUrl)
    return adapter.validate_python(str_url)


def to_urls(str_urls: List[str]) -> List[HttpUrl]:
    adapter = TypeAdapter(List[HttpUrl])
    return adapter.validate_python(str_urls)


class ThebeDatasetStrURLs(BaseModel):
    fault_test: List[str] = [
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862483&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862488&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862497&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862485&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862496&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862492&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862486&version=4.0",
    ]
    fault_train: List[str] = [
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862484&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862490&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862491&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862489&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862499&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862487&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862493&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862495&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862500&version=4.0",
    ]
    fault_val: List[str] = [
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862498&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862494&version=4.0",
    ]
    seis_test: List[str] = [
        "https://dataverse.harvard.edu/file.xhtml?fileId=4863110&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4863111&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4863109&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4863126&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4863125&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4863123&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4863124&version=4.0",
    ]
    seis_train: List[str] = [
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862642&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862655&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862656&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862781&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862788&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862793&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4862823&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4863049&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4863068&version=4.0",
    ]
    seis_val: List[str] = [
        "https://dataverse.harvard.edu/file.xhtml?fileId=4863099&version=4.0",
        "https://dataverse.harvard.edu/file.xhtml?fileId=4863098&version=4.0",
    ]

    class Config:
        frozen = True


thebe_dataset_str_urls = ThebeDatasetStrURLs()


class ThebeDatasetURLs(BaseModel):
    fault_test: List[HttpUrl] = to_urls(thebe_dataset_str_urls.fault_test)
    fault_train: List[HttpUrl] = to_urls(thebe_dataset_str_urls.fault_train)
    fault_val: List[HttpUrl] = to_urls(thebe_dataset_str_urls.fault_val)
    seis_test: List[HttpUrl] = to_urls(thebe_dataset_str_urls.seis_test)
    seis_train: List[HttpUrl] = to_urls(thebe_dataset_str_urls.seis_train)
    seis_val: List[HttpUrl] = to_urls(thebe_dataset_str_urls.seis_val)

    class Config:
        frozen = True


thebe_dataset_urls = ThebeDatasetURLs()


def convert_dataset_url_to_download_url(dataset_url: HttpUrl) -> HttpUrl:
    # dataset  url: https://dataverse.harvard.edu/file.xhtml?fileId=${ID}&version=4.0
    # download url: https://dataverse.harvard.edu/api/access/datafile/${ID}?gbrecs=true

    query = dataset_url.query

    id_start = query.find("fileId=") + len("fileId=")
    id_end = query.find("&", id_start)
    id_value = query[id_start:id_end]

    download_url = to_url(f"https://dataverse.harvard.edu/api/access/datafile/{id_value}?gbrecs=true")

    return download_url


working_dir = datasets_root_path.joinpath("thebe")

download_url = convert_dataset_url_to_download_url(thebe_dataset_urls.seis_test[0])
cmd = f"wget -q {download_url}"
print(cmd)
# subprocess.run(, shell=True)
