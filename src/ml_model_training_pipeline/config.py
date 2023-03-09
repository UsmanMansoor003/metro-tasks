import json
from dataclasses import dataclass
from typing import List

import pkg_resources
from loguru import logger


@dataclass
class Config:
    csv_url: str
    col_names: List[str]
    table_name: str

    @classmethod
    def get(cls):
        conf_dict = json.loads(
            pkg_resources.resource_string(__name__, f"resources/config.json")
        )
        logger.info(f"Using config {conf_dict}")
        return cls(**conf_dict)
