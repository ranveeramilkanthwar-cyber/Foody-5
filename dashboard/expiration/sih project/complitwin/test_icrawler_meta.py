import os
import json
import csv
import hashlib
import io
from pathlib import Path
from PIL import Image
from icrawler.builtin import BingImageCrawler
from icrawler import ImageDownloader

class CustomDownloader(ImageDownloader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.downloaded_meta = []

    def download(self, task, default_ext, timeout=5, max_retry=3, overwrite=False, **kwargs):
        file_url = task['file_url']
        return super().download(task, default_ext, timeout=timeout, max_retry=max_retry, overwrite=overwrite, **kwargs)

crawler = BingImageCrawler(
    downloader_cls=CustomDownloader,
    storage={'root_dir': 'tmp_test_meta'}
)
crawler.crawl(keyword='Parle-G biscuit packet back label declaration', max_num=3)
print("Done crawling with custom downloader")
