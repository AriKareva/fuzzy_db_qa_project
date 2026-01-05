from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import URL
from datetime import datetime
from venv import logger

class DBConnector:
    SUPPORTED_DB_TYPES = {
        'postgresql+psycopg2': 'PostgreSQL',
    }

    DEFAULT_POOL_CONFIG = {
        'pool_size': 5,
        'max_overflow': 10,
        'pool_timeout': 30,
        'pool_recycle': 3600
    }

    MAX_RETRIES = 3
    RETRY_DELAY = 2 

    host: str
    port: int
    user_name: str
    password: str
    dbname: str
    engine_name: str
    engine: Optional[Engine]

    def __init__(self, host: str, port: int, user_name: str, password: str, dbname: str, engine_name: str):
        self.host = host
        self.port = int(port)
        self.user_name = user_name
        self.password = password
        self.dbname = dbname
        self.engine_name = engine_name
        self.engine = None
        self._last_connection_time: Optional[datetime] = None

    def validate_credentials(self) -> bool:
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("host должен быть непустой строкой.")
        if not isinstance(self.port, int) or not (1 <= self.port <= 65535):
            raise ValueError("port должен быть целым числом в диапазоне 1-65535.")
        if not isinstance(self.user_name, str) or not self.user_name.strip():
            raise ValueError("username должен быть непустой строкой.")
        if not isinstance(self.dbname, str) or not self.dbname.strip():
            raise ValueError("dbname должен быть непустой строкой.")
        if self.engine_name not in self.SUPPORTED_DB_TYPES:
            raise ValueError(f"engine_name '{self.engine_name}' не поддерживается. Поддержаны: {list(self.SUPPORTED_DB_TYPES.keys())}")
        return True

    def set_connection(self) -> bool:
        self.validate_credentials()

        connection_data = URL.create(
            drivername=self.engine_name,
            username=self.user_name,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.dbname
        )

        pool_cfg = self.DEFAULT_POOL_CONFIG.copy()

        try:
            engine = create_engine(
                connection_data,
                pool_size=pool_cfg['pool_size'],
                max_overflow=pool_cfg['max_overflow'],
                pool_timeout=pool_cfg['pool_timeout'],
                echo=False,
                pool_recycle=pool_cfg['pool_recycle']
            )

            with engine.connect() as db_connection:
                init_query_result = db_connection.execute(text('SELECT 1;'))
                self.engine = engine
                self._last_connection_time = datetime.utcnow()
                return True

        except SQLAlchemyError as err:
            logger.error(f'Connection error! {err}')
            raise ConnectionError(f"Не удалось подключиться к БД: {err}")

    def disconnect(self) -> None:
        try:
            if self.engine is not None:
                self.engine.dispose()
        finally:
            self.engine = None
            self._last_connection_time = None


    def get_engine(self) -> Engine:
        if self.engine is None:
            raise RuntimeError("Нет активного подключения.")
        return self.engine
