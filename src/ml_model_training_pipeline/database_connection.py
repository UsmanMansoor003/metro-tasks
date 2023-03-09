from dataclasses import dataclass, field
import os
from loguru import logger
from sqlalchemy import create_engine, Engine


@dataclass
class Postgres:
    conn: Engine = field(default=None)

    @staticmethod
    def _create_connection() -> Engine:
        db_url = os.environ.get('DB_URL')
        return create_engine(db_url)

    def __post_init__(self):
        self.conn = self._create_connection()
        logger.info(
            f"Open Connection to postgres data: {self.conn}"
        )
