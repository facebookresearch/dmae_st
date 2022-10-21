#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import copy
import sys
import tempfile
from io import BytesIO

import av

# import common.thread_safe_fork
import pycurl
# import ti.urlgen.everstore_url_py as everstore_url_py

# common.thread_safe_fork.threadSafeForkRegisterAtFork()


# def get_everstore_urls(handles):
#     assert len(handles) == 1
#     sys.argv = ["."]
#     try:
#         c = everstore_url_py.EverstoreCdnUrlPyClientWrapper()
#         url = c.buildInterncacheUrl("fair/cv", handles[0])
#         return ["" if "/0_0_0_\x00.jpg" in url else url]
#     except Exception:
#         print("failed to get data by handle {}".format(handles))
#         return None


def get_curl_handles(num_handles):
    multi = pycurl.CurlMulti()
    multi.handles = [None] * num_handles
    for idx in range(num_handles):
        handle = pycurl.Curl()
        handle.fp = None
        handle.setopt(pycurl.FOLLOWLOCATION, 1)
        handle.setopt(pycurl.MAXREDIRS, 5)
        handle.setopt(pycurl.CONNECTTIMEOUT, 15)
        handle.setopt(pycurl.TIMEOUT, 10)
        handle.setopt(pycurl.NOSIGNAL, 1)
        multi.handles[idx] = handle
    return multi


def download(multi, urls, num_retries=2):  # noqa
    # repeat until all images are processed:
    handles = multi.handles[:]
    urls_to_request = copy.deepcopy(urls)
    num_urls, num_downloaded = len(urls), 0
    urls_to_request = [(ind, urls_to_request[ind]) for ind in range(num_urls)]
    success = [False] * num_urls
    retries = [0] * num_urls
    samples = [None] * num_urls
    while num_downloaded < num_urls:
        # add download handles to the curl request:
        while urls_to_request and handles:
            idxinlist, url = urls_to_request.pop()
            if url == "":
                samples[idxinlist] = None
                num_downloaded += 1
            else:
                handle = handles.pop()
                handle.fp = BytesIO()
                handle.setopt(pycurl.URL, url)
                handle.setopt(pycurl.WRITEFUNCTION, handle.fp.write)
                multi.add_handle(handle)
                handle.index = idxinlist  # list index in which to store image

        while True:
            status, _ = multi.perform()
            if status != pycurl.E_CALL_MULTI_PERFORM:
                break

        while True:
            num, oklist, errlist = multi.info_read()

            for handle in oklist:
                data = handle.fp.getvalue()
                if data != b"Content not found" and data != b"Gone":
                    try:
                        samples[handle.index] = BytesIO(data)
                        success[handle.index] = True
                    except Exception:
                        pass
                else:
                    samples[handle.index] = None
                    success[handle.index] = False

            for (handle, _, _) in errlist:
                samples[handle.index] = None
                handle.fp.close()
                # logger.error('Download failed. %s' % msg)

            for handle in oklist + [tup[0] for tup in errlist]:
                handle_succeeded = success[handle.index]
                if handle_succeeded or retries[handle.index] >= num_retries:
                    num_downloaded += 1
                else:
                    retries[handle.index] += 1
                    urls_to_request.append((handle.index, urls[handle.index]))
                handle.fp = None
                multi.remove_handle(handle)
                handles.append(handle)
            if num == 0:
                break
    del urls_to_request
    del handles
    return samples, success


def get_video_container(handle, multi_thread_decode=False,
                        backend="torchvision"):
    if backend == "torchvision": # use torchvision by default
        with open(handle, "rb") as fp:
            container = fp.read()
        return container
    elif backend == "pyav": # use pyav if stated
        container = av.open(handle)
        if multi_thread_decode:
            container.streams.video[0].thread_type = "AUTO"
        return container
    return None
