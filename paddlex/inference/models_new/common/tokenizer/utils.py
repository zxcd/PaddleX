# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import stat
import uuid
import copy
import time
import tqdm
import inspect
import requests
import tempfile
import threading

from functools import lru_cache
from urllib.parse import urlparse
from dataclasses import dataclass
from argparse import ArgumentTypeError
from pathlib import Path
from typing import List, Dict, Literal, Optional, Union, Callable, BinaryIO

from lazy_paddle import __version__

from filelock import FileLock
from contextlib import contextmanager
from functools import partial
from paddlex.utils import logging

logger = logging.getLogger(__name__)

__all__ = [
    "OfflineModeIsEnabled",
    "OfflineAdapter",
    "AistudioBosFileMetadata",
]


def _get_user_home():
    return os.path.expanduser("~")


def _get_ppnlp_home():
    if "PPNLP_HOME" in os.environ:
        home_path = os.environ["PPNLP_HOME"]
        if os.path.exists(home_path):
            if os.path.isdir(home_path):
                return home_path
            else:
                raise RuntimeError(
                    "The environment variable PPNLP_HOME {} is not a directory.".format(
                        home_path
                    )
                )
        else:
            return home_path
    return os.path.join(_get_user_home(), ".paddlenlp")


def _get_sub_home(directory, parent_home=_get_ppnlp_home()):
    home = os.path.join(parent_home, directory)
    if not os.path.exists(home):
        os.makedirs(home, exist_ok=True)
    return home


MODEL_HOME = _get_sub_home("models")
ENDPOINT = os.getenv("PPNLP_ENDPOINT", "https://bj.bcebos.com/paddlenlp")
ENDPOINT_v2 = "https://paddlenlp.bj.bcebos.com"

BOS_URL_TEMPLATE = ENDPOINT + "/{repo_type}/community/{repo_id}/{revision}/{filename}"
BOS_URL_TEMPLATE_WITHOUT_REVISION = (
    ENDPOINT + "/{repo_type}/community/{repo_id}/{filename}"
)


REGEX_COMMIT_HASH = re.compile(r"^[0-9a-f]{40}$")
REPO_TYPE = "models"
DEFAULT_ETAG_TIMEOUT = 10
DEFAULT_REQUEST_TIMEOUT = 10
DEFAULT_DOWNLOAD_TIMEOUT = 10
DOWNLOAD_CHUNK_SIZE = 10 * 1024 * 1024
HEADER_FILENAME_PATTERN = re.compile(r'filename="(?P<filename>.*?)"')

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}


def _is_true(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.upper() in ENV_VARS_TRUE_VALUES


def _as_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def strtobool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def _chmod_and_replace(src: str, dst: str) -> None:
    """Set correct permission before moving a blob from tmp directory to cache dir.

    Do not take into account the `umask` from the process as there is no convenient way
    to get it that is thread-safe.

    See:
    - About umask: https://docs.python.org/3/library/os.html#os.umask
    - Thread-safety: https://stackoverflow.com/a/70343066
    - About solution: https://github.com/huggingface/huggingface_hub/pull/1220#issuecomment-1326211591
    - Fix issue: https://github.com/huggingface/huggingface_hub/issues/1141
    - Fix issue: https://github.com/huggingface/huggingface_hub/issues/1215
    """
    # Get umask by creating a temporary file in the cached repo folder.
    tmp_file = Path(dst).parent.parent / f"tmp_{uuid.uuid4()}"
    try:
        tmp_file.touch()
        cache_dir_mode = Path(tmp_file).stat().st_mode
        os.chmod(src, stat.S_IMODE(cache_dir_mode))
    finally:
        tmp_file.unlink()


def http_get(
    url: str,
    temp_file: BinaryIO,
    *,
    proxies=None,
    resume_size: float = 0,
    headers: Optional[Dict[str, str]] = None,
    expected_size: Optional[int] = None,
    _nb_retries: int = 5,
):
    """
    Download a remote file. Do not gobble up errors, and will return errors tailored to the Hugging Face Hub.

    If ConnectionError (SSLError) or ReadTimeout happen while streaming data from the server, it is most likely a
    transient error (network outage?). We log a warning message and try to resume the download a few times before
    giving up. The method gives up after 5 attempts if no new data has being received from the server.
    """
    initial_headers = headers
    headers = copy.deepcopy(headers) or {}
    if resume_size > 0:
        headers["Range"] = "bytes=%d-" % (resume_size,)

    r = _request_wrapper(
        method="GET",
        url=url,
        stream=True,
        proxies=proxies,
        headers=headers,
        timeout=DEFAULT_DOWNLOAD_TIMEOUT,
    )
    raise_for_status(r)
    content_length = r.headers.get("Content-Length")

    # NOTE: 'total' is the total number of bytes to download, not the number of bytes in the file.
    #       If the file is compressed, the number of bytes in the saved file will be higher than 'total'.
    total = resume_size + int(content_length) if content_length is not None else None

    displayed_name = url
    content_disposition = r.headers.get("Content-Disposition")
    if content_disposition is not None:
        match = HEADER_FILENAME_PATTERN.search(content_disposition)
        if match is not None:
            # Means file is on CDN
            displayed_name = match.groupdict()["filename"]

    # Truncate filename if too long to display
    if len(displayed_name) > 40:
        displayed_name = f"(…){displayed_name[-40:]}"

    consistency_error_message = (
        f"Consistency check failed: file should be of size {expected_size} but has size"
        f" {{actual_size}} ({displayed_name}).\nWe are sorry for the inconvenience. Please retry download and"
        " pass `force_download=True, resume_download=False` as argument.\nIf the issue persists, please let us"
        " know by opening an issue on https://github.com/huggingface/huggingface_hub."
    )

    # Stream file to buffer
    with tqdm(
        unit="B",
        unit_scale=True,
        total=total,
        initial=resume_size,
        desc=displayed_name,
        disable=bool(logger.getEffectiveLevel() == logging.NOTSET),
    ) as progress:
        new_resume_size = resume_size
        try:
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    temp_file.write(chunk)
                    new_resume_size += len(chunk)
                    # Some data has been downloaded from the server so we reset the number of retries.
                    _nb_retries = 5
        except (requests.ConnectionError, requests.ReadTimeout) as e:
            # If ConnectionError (SSLError) or ReadTimeout happen while streaming data from the server, it is most likely
            # a transient error (network outage?). We log a warning message and try to resume the download a few times
            # before giving up. Tre retry mechanism is basic but should be enough in most cases.
            if _nb_retries <= 0:
                logger.warning(
                    "Error while downloading from %s: %s\nMax retries exceeded.",
                    url,
                    str(e),
                )
                raise
            logger.warning(
                "Error while downloading from %s: %s\nTrying to resume download...",
                url,
                str(e),
            )
            time.sleep(1)
            reset_sessions()  # In case of SSLError it's best to reset the shared requests.Session objects
            return http_get(
                url=url,
                temp_file=temp_file,
                proxies=proxies,
                resume_size=new_resume_size,
                headers=initial_headers,
                expected_size=expected_size,
                _nb_retries=_nb_retries - 1,
            )

        if expected_size is not None and expected_size != temp_file.tell():
            raise EnvironmentError(
                consistency_error_message.format(
                    actual_size=temp_file.tell(),
                )
            )


class OfflineModeIsEnabled(requests.ConnectionError):
    """Raised when a request is made but `AISTUDIO_HUB_OFFLINE=1` is set as environment variable."""


class OfflineAdapter(requests.adapters.HTTPAdapter):
    def send(
        self, request: requests.models.PreparedRequest, *args, **kwargs
    ) -> requests.Response:
        raise OfflineModeIsEnabled(
            f"Cannot reach {request.url}: offline mode is enabled. To disable it, please unset the `AISTUDIO_HUB_OFFLINE` environment variable."
        )


def raise_for_status(
    response: requests.Response, endpoint_name: Optional[str] = None
) -> None:
    try:
        response.raise_for_status()
    except response.HTTPError as e:
        if response.status_code == 404:
            message = (
                f"{response.status_code} Client Error."
                + "\n\n"
                + f"Entry Not Found for url: {response.url}."
            )
        elif response.status_code == 400:
            message = (
                f"\n\nBad request for {endpoint_name} endpoint:"
                if endpoint_name is not None
                else "\n\nBad request:"
            )


def bos_url(
    repo_id: str,
    filename: str,
    *,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> str:
    if subfolder == "":
        subfolder = None
    if subfolder is not None:
        filename = f"{subfolder}/{filename}"

    url = BOS_URL_TEMPLATE_WITHOUT_REVISION.format(
        repo_type=REPO_TYPE,
        repo_id=repo_id,
        filename=filename,
    )

    # Update endpoint if provided
    if endpoint is not None and url.startswith(ENDPOINT):
        url = endpoint + url[len(ENDPOINT) :]
    return url


def bos_download(
    repo_id: str = None,
    filename: str = None,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    local_dir: Union[str, Path, None] = None,
    local_dir_use_symlinks: Union[bool, Literal["auto"]] = "auto",
    user_agent: Union[Dict, str, None] = None,
    force_download: bool = False,
    proxies: Optional[Dict] = None,
    etag_timeout: float = DEFAULT_ETAG_TIMEOUT,
    resume_download: bool = False,
    token: Optional[str] = None,
    local_files_only: bool = False,
    endpoint: Optional[str] = None,
    url: Optional[str] = None,
    **kwargs,
):
    if url is not None:
        if repo_id is None:
            if url.startswith(ENDPOINT):
                repo_id = "/".join(url[len(ENDPOINT) + 1 :].split("/")[:-1])
            else:
                repo_id = "/".join(url[len(ENDPOINT_v2) + 1 :].split("/")[:-1])
        if filename is None:
            filename = url.split("/")[-1]
        subfolder = None

    if cache_dir is None:
        cache_dir = MODEL_HOME
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if subfolder == "":
        subfolder = None
    if subfolder is not None:
        # This is used to create a URL, and not a local path, hence the forward slash.
        filename = f"{subfolder}/{filename}"

    locks_dir = os.path.join(cache_dir, ".locks")

    storage_folder = os.path.join(cache_dir, repo_id)
    os.makedirs(storage_folder, exist_ok=True)

    if url is None:
        url = bos_url(repo_id, filename, repo_type=REPO_TYPE, endpoint=endpoint)
    headers = None
    url_to_download = url
    lock_path = os.path.join(locks_dir, repo_id, f"{filename}.lock")
    file_path = os.path.join(cache_dir, repo_id, filename)

    if os.name == "nt" and len(os.path.abspath(lock_path)) > 255:
        lock_path = "\\\\?\\" + os.path.abspath(lock_path)

    if os.name == "nt" and len(os.path.abspath(file_path)) > 255:
        file_path = "\\\\?\\" + os.path.abspath(file_path)

    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_path):
        # If the download just completed while the lock was activated.
        if os.path.exists(file_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return file_path

        if resume_download:
            incomplete_path = file_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager():
                with open(incomplete_path, "ab") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(  # type: ignore
                tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False
            )
            resume_size = 0

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logger.info("downloading %s to %s", url_to_download, temp_file.name)

            http_get(
                url_to_download,
                temp_file,
                proxies=proxies,
                resume_size=resume_size,
                headers=headers,
            )

        logger.info("storing %s in cache at %s", url_to_download, file_path)
        _chmod_and_replace(temp_file.name, file_path)
    try:
        os.remove(lock_path)
    except OSError:
        pass
    return file_path


def resolve_file_path(
    repo_id: str = None,
    filenames: Union[str, list] = None,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    library_name: Optional[str] = "PaddleNLP",
    library_version: Optional[str] = __version__,
    cache_dir: Union[str, Path, None] = None,
    local_dir: Union[str, Path, None] = None,
    local_dir_use_symlinks: Union[bool, Literal["auto"]] = "auto",
    user_agent: Union[Dict, str, None] = None,
    force_download: bool = False,
    proxies: Optional[Dict] = None,
    etag_timeout: float = 10,
    resume_download: bool = False,
    token: Union[bool, str, None] = None,
    local_files_only: bool = False,
    endpoint: Optional[str] = None,
    url: Optional[str] = None,
    from_aistudio: bool = False,
    from_hf_hub: bool = False,
    from_bos: bool = True,
) -> str:
    """
    This is a general download function, mainly called by the from_pretrained function.

    It supports downloading files from four different download sources, including BOS, AiStudio,
    HuggingFace Hub and ModelScope.

    If you want to download a file from ModelScope, you need to set os.environ["from_modelscope"] = "True"

    Args:
        repo_id('str'): A path to a folder containing the file, a path of the file, a url or repo name.
        filenames('str' or list): Name of the file to be downloaded. If it is a str, the file will be downloaded directly,
            if it is a list, it will try to download the file in turn, and when one exists, it will be returned directly.
        subfolder('str'): Some repos will exist subfolder.
        repo_type('str'): The default is model.
        cache_dir('str' or Path): Where to save or load the file after downloading.
        url('str'): If it is not None, then it will be downloaded from BOS.
        from_aistudio('bool'): If this value is true, it will be downloaded from aistudio.
        from_hf_hub('bool'): If this value is true, it will be downloaded from hf hub.
        from_bos('bool'): If this value is true, it will be downloaded from bos (default).


    Returns:
        cached_file('str'): The path of file or None.
    """
    assert repo_id is not None, "repo_id cannot be None"
    assert filenames is not None, "filenames cannot be None"

    if isinstance(filenames, str):
        filenames = [filenames]

    download_kwargs = dict(
        repo_id=repo_id,
        filename=filenames[0],
        subfolder=subfolder if subfolder is not None else "",
        repo_type=repo_type,
        revision=revision,
        library_name=library_name,
        library_version=library_version,
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_dir_use_symlinks=local_dir_use_symlinks,
        user_agent=user_agent,
        force_download=force_download,
        proxies=proxies,
        etag_timeout=etag_timeout,
        resume_download=resume_download,
        token=token,
        local_files_only=local_files_only,
        endpoint=endpoint,
    )
    cached_file = None
    log_endpoint = "N/A"
    # log_filename = os.path.join(download_kwargs["subfolder"], filename)

    # return file path from local file, eg: /cache/path/model_config.json
    if os.path.isfile(repo_id):
        return repo_id
    # return the file path from local dir with filename, eg: /local/path
    elif os.path.isdir(repo_id):
        for index, filename in enumerate(filenames):
            if os.path.exists(
                os.path.join(repo_id, download_kwargs["subfolder"], filename)
            ):
                if not os.path.isfile(
                    os.path.join(repo_id, download_kwargs["subfolder"], filename)
                ):
                    raise EnvironmentError(
                        f"{repo_id} does not appear to have file named {filename}."
                    )
                return os.path.join(repo_id, download_kwargs["subfolder"], filename)
            elif index < len(filenames) - 1:
                continue
            else:
                raise FileNotFoundError(
                    f"please make sure one of the {filenames} under the dir {repo_id}"
                )

    # check cache
    for filename in filenames:
        cache_file_name = bos_aistudio_hf_try_to_load_from_cache(
            repo_id,
            filename,
            cache_dir,
            subfolder,
            revision,
            repo_type,
            from_bos,
            from_aistudio,
            from_hf_hub,
        )
        if cache_file_name is not None:
            return cache_file_name

    # download file from different origins
    try:
        if filenames[0].startswith("http://") or filenames[0].startswith("https://"):
            log_endpoint = "BOS"
            download_kwargs["url"] = filenames[0]
            download_kwargs["repo_id"] = repo_id
            if filenames[0].split("/")[-1].endswith("pdparams"):
                download_kwargs["filename"] = "model_state.pdparams"
            else:
                download_kwargs["filename"] = None
            cached_file = bos_download(
                **download_kwargs,
            )
            return cached_file
        else:
            log_endpoint = "BOS"
            download_kwargs["url"] = url
            for filename in filenames:
                download_kwargs["filename"] = filename
                is_available = bos_aistudio_hf_file_exist(
                    repo_id,
                    filename,
                    subfolder=subfolder,
                    repo_type=repo_type,
                    revision=revision,
                    token=token,
                    endpoint=endpoint,
                    from_bos=from_bos,
                    from_aistudio=from_aistudio,
                    from_hf_hub=from_hf_hub,
                )
                if is_available:
                    cached_file = bos_download(
                        **download_kwargs,
                    )
                    if cached_file is not None:
                        return cached_file
    except requests.HTTPError as err:
        raise EnvironmentError(
            f"There was a specific connection error when trying to load {repo_id}:\n{err}"
        )
    except ValueError:
        raise EnvironmentError(
            f"We couldn't connect to '{log_endpoint}' to load this model, couldn't find it"
            f" in the cached files and it looks like {repo_id} is not the path to a"
            f" directory containing one of the {filenames} or"
            " \nCheckout your internet connection or see how to run the library in offline mode."
        )
    except EnvironmentError:
        raise EnvironmentError(
            f"Can't load the model for '{repo_id}'. If you were trying to load it from "
            f"'{log_endpoint}', make sure you don't have a local directory with the same name. "
            f"Otherwise, make sure '{repo_id}' is the correct path to a directory "
            f"containing one of the {filenames}"
        )


BACKEND_FACTORY_T = Callable[[], requests.Session]
OFFLINE = _is_true(os.environ.get("AISTUDIO_BOS_OFFLINE"))


def _default_backend_factory() -> requests.Session:
    session = requests.Session()
    if OFFLINE:
        session.mount("http://", OfflineAdapter())
        session.mount("https://", OfflineAdapter())

    return session


_GLOBAL_BACKEND_FACTORY: BACKEND_FACTORY_T = _default_backend_factory
HTTP_METHOD_T = Literal["GET", "OPTIONS", "HEAD", "POST", "PUT", "PATCH", "DELETE"]


@lru_cache
def _get_session_from_cache(process_id: int, thread_id: int) -> requests.Session:
    """
    Create a new session per thread using global factory. Using LRU cache (maxsize 128) to avoid memory leaks when
    using thousands of threads. Cache is cleared when `configure_http_backend` is called.
    """
    return _GLOBAL_BACKEND_FACTORY()


def reset_sessions() -> None:
    """Reset the cache of sessions.

    Mostly used internally when sessions are reconfigured or an SSLError is raised.
    See [`configure_http_backend`] for more details.
    """
    _get_session_from_cache.cache_clear()


def get_session() -> requests.Session:
    """
    Get a `requests.Session` object, using the session factory from the user.

    Use [`get_session`] to get a configured Session. Since `requests.Session` is not guaranteed to be thread-safe,
    `huggingface_hub` creates 1 Session instance per thread. They are all instantiated using the same `backend_factory`
    set in [`configure_http_backend`]. A LRU cache is used to cache the created sessions (and connections) between
    calls. Max size is 128 to avoid memory leaks if thousands of threads are spawned.

    See [this issue](https://github.com/psf/requests/issues/2766) to know more about thread-safety in `requests`.

    Example:
    ```py
    import requests
    from huggingface_hub import configure_http_backend, get_session

    # Create a factory function that returns a Session with configured proxies
    def backend_factory() -> requests.Session:
        session = requests.Session()
        session.proxies = {"http": "http://10.10.1.10:3128", "https": "https://10.10.1.11:1080"}
        return session

    # Set it as the default session factory
    configure_http_backend(backend_factory=backend_factory)

    # In practice, this is mostly done internally in `huggingface_hub`
    session = get_session()
    ```
    """
    return _get_session_from_cache(
        process_id=os.getpid(), thread_id=threading.get_ident()
    )


def _normalize_etag(etag: Optional[str]) -> Optional[str]:
    """Normalize ETag HTTP header, so it can be used to create nice filepaths.

    The HTTP spec allows two forms of ETag:
      ETag: W/"<etag_value>"
      ETag: "<etag_value>"

    For now, we only expect the second form from the server, but we want to be future-proof so we support both. For
    more context, see `TestNormalizeEtag` tests and https://github.com/huggingface/huggingface_hub/pull/1428.

    Args:
        etag (`str`, *optional*): HTTP header

    Returns:
        `str` or `None`: string that can be used as a nice directory name.
        Returns `None` if input is None.
    """
    if etag is None:
        return None
    return etag.lstrip("W/").strip('"')


def _request_wrapper(
    method: HTTP_METHOD_T,
    url: str,
    *,
    follow_relative_redirects: bool = False,
    **params,
) -> requests.Response:
    """Wrapper around requests methods to follow relative redirects if `follow_relative_redirects=True` even when
    `allow_redirection=False`.

    Args:
        method (`str`):
            HTTP method, such as 'GET' or 'HEAD'.
        url (`str`):
            The URL of the resource to fetch.
        follow_relative_redirects (`bool`, *optional*, defaults to `False`)
            If True, relative redirection (redirection to the same site) will be resolved even when `allow_redirection`
            kwarg is set to False. Useful when we want to follow a redirection to a renamed repository without
            following redirection to a CDN.
        **params (`dict`, *optional*):
            Params to pass to `requests.request`.
    """
    # Recursively follow relative redirects
    if follow_relative_redirects:
        response = _request_wrapper(
            method=method,
            url=url,
            follow_relative_redirects=False,
            **params,
        )

        # If redirection, we redirect only relative paths.
        # This is useful in case of a renamed repository.
        if 300 <= response.status_code <= 399:
            parsed_target = urlparse(response.headers["Location"])
            if parsed_target.netloc == "":
                # This means it is a relative 'location' headers, as allowed by RFC 7231.
                # (e.g. '/path/to/resource' instead of 'http://domain.tld/path/to/resource')
                # We want to follow this relative redirect !
                #
                # Highly inspired by `resolve_redirects` from requests library.
                # See https://github.com/psf/requests/blob/main/requests/sessions.py#L159
                next_url = urlparse(url)._replace(path=parsed_target.path).geturl()
                return _request_wrapper(
                    method=method,
                    url=next_url,
                    follow_relative_redirects=True,
                    **params,
                )
        return response
    # Perform request and return if status_code is not in the retry list.
    response = get_session().request(method=method, url=url, **params)
    raise_for_status(response)
    return response


def get_bos_file_metadata(
    url: str,
    token: Union[bool, str, None] = None,
    proxies: Optional[Dict] = None,
    timeout: Optional[float] = DEFAULT_REQUEST_TIMEOUT,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    user_agent: Union[Dict, str, None] = None,
):
    """Fetch metadata of a file versioned on the Hub for a given url.

    Args:
        url (`str`):
            File url, for example returned by [`bos_url`].
        token (`str` or `bool`, *optional*):
            A token to be used for the download.
                - If `True`, the token is read from the BOS config
                  folder.
                - If `False` or `None`, no token is provided.
                - If a string, it's used as the authentication token.
        proxies (`dict`, *optional*):
            Dictionary mapping protocol to the URL of the proxy passed to
            `requests.request`.
        timeout (`float`, *optional*, defaults to 10):
            How many seconds to wait for the server to send metadata before giving up.
        library_name (`str`, *optional*):
            The name of the library to which the object corresponds.
        library_version (`str`, *optional*):
            The version of the library.
        user_agent (`dict`, `str`, *optional*):
            The user-agent info in the form of a dictionary or a string.

    Returns:
        A [`AistudioBosFileMetadata`] object containing metadata such as location, etag, size and
        commit_hash.
    """
    headers = {}
    headers["Accept-Encoding"] = (
        "identity"  # prevent any compression => we want to know the real size of the file
    )

    # Retrieve metadata
    r = _request_wrapper(
        method="HEAD",
        url=url,
        headers=headers,
        allow_redirects=False,
        follow_relative_redirects=True,
        proxies=proxies,
        timeout=timeout,
    )
    raise_for_status(r)

    # Return
    return AistudioBosFileMetadata(
        commit_hash=None,
        etag=_normalize_etag(r.headers.get("ETag")),
        location=url,
        size=_as_int(r.headers.get("Content-Length")),
    )


@dataclass(frozen=True)
class AistudioBosFileMetadata:
    """Data structure containing information about a file versioned on the Aistudio Hub.

    Returned by [`get_aistudio_file_metadata`] based on a URL.

    Args:
        commit_hash (`str`, *optional*):
            The commit_hash related to the file.
        etag (`str`, *optional*):
            Etag of the file on the server.
        location (`str`):
            Location where to download the file. Can be a Hub url or not (CDN).
        size (`size`):
            Size of the file. In case of an LFS file, contains the size of the actual
            LFS file, not the pointer.
    """

    commit_hash: Optional[str]
    etag: Optional[str]
    location: str
    size: Optional[int]


def bos_file_exists(
    repo_id: str,
    filename: str,
    *,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> bool:
    url = bos_url(
        repo_id=repo_id, repo_type=REPO_TYPE, filename=filename, endpoint=endpoint
    )
    try:
        get_bos_file_metadata(url, token=token)
        return True
    except OSError:
        return False


def bos_aistudio_hf_file_exist(
    repo_id: str,
    filename: str,
    *,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    endpoint: Optional[str] = None,
    from_bos: bool = True,
    from_aistudio: bool = False,
    from_hf_hub: bool = False,
):
    assert repo_id is not None, "repo_id cannot be None"
    assert filename is not None, "filename cannot be None"

    if subfolder is None:
        subfolder = ""
    filename = os.path.join(subfolder, filename)
    out = bos_file_exists(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
        token=token,  # donot need token
        endpoint=endpoint,
    )
    return out


def bos_try_to_load_from_cache(
    repo_id: str,
    filename: str,
    cache_dir: Union[str, Path, None] = None,
    revision: Optional[str] = None,
    repo_type: Optional[str] = None,
):
    if cache_dir is None:
        cache_dir = MODEL_HOME

    cached_file = os.path.join(cache_dir, repo_id, filename)
    return cached_file if os.path.isfile(cached_file) else None


def bos_aistudio_hf_try_to_load_from_cache(
    repo_id: str,
    filename: str,
    cache_dir: Union[str, Path, None] = None,
    subfolder: str = None,
    revision: Optional[str] = None,
    repo_type: Optional[str] = None,
    from_bos: bool = True,
    from_aistudio: bool = False,
    from_hf_hub: bool = False,
):
    if subfolder is None:
        subfolder = ""
    load_kwargs = dict(
        repo_id=repo_id,
        filename=os.path.join(subfolder, filename),
        cache_dir=cache_dir,
        revision=revision,
        repo_type=repo_type,
    )
    return bos_try_to_load_from_cache(**load_kwargs)


def convert_to_dict_message(conversation: List[List[str]]):
    """Convert the list of chat messages to a role dictionary chat messages."""
    conversations = []
    for index, item in enumerate(conversation):
        assert (
            1 <= len(item) <= 2
        ), "Each Rounds in conversation should have 1 or 2 elements."
        if isinstance(item[0], str):
            conversations.append({"role": "user", "content": item[0]})
            if len(item) == 2 and isinstance(item[1], str):
                conversations.append({"role": "assistant", "content": item[1]})
            else:
                # If there is only one element in item, it must be the last round.
                # If it is not the last round, it must be an error.
                if index != len(conversation) - 1:
                    raise ValueError(f"Round {index} has error round")
        else:
            raise ValueError("Each round in list should be string")
    return conversations


def fn_args_to_dict(func, *args, **kwargs):
    """
    Inspect function `func` and its arguments for running, and extract a
    dict mapping between argument names and keys.
    """
    if hasattr(inspect, "getfullargspec"):
        (spec_args, spec_varargs, spec_varkw, spec_defaults, _, _, _) = (
            inspect.getfullargspec(func)
        )
    else:
        (spec_args, spec_varargs, spec_varkw, spec_defaults) = inspect.getargspec(func)
    # add positional argument values
    init_dict = dict(zip(spec_args, args))
    # add default argument values
    kwargs_dict = (
        dict(zip(spec_args[-len(spec_defaults) :], spec_defaults))
        if spec_defaults
        else {}
    )
    for k in list(kwargs_dict.keys()):
        if k in init_dict:
            kwargs_dict.pop(k)
    kwargs_dict.update(kwargs)
    init_dict.update(kwargs_dict)
    return init_dict
