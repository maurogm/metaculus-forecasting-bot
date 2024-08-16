"""Módulo para administrar los loggers"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
# TODO: Make singleton
class LoggerFactory:
    """Factory para la creción de loggers.

    Parameters
    ----------
    log_level : str
        Nivel de log a usar. Puede ser DEBUG, INFO, WARNING, ERROR o CRITICAL.
    log_to_console : bool
        Si se debe loggear a la consola.
    logs_file_dir : Optional[str]
        Directorio donde se debe guardar el archivo de logs. Si es None, no se loguea a disco.
    logs_file_name : Optional[str]
        Nombre del archivo de logs. Si es None, no se loguea a disco.
    """

    log_level: str = "INFO"
    log_to_console: bool = True
    logs_file_dir: Optional[str] = None
    logs_file_name: Optional[str] = None

    formatter = logging.Formatter(
        fmt="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %Z",
    )


    def make_logger(self, name: str, level: Optional[str] = None):
        """Crea un logger con el nombre especificado.

        Si se pasa un nivel de log explícito, se usa ese en vez del default.
        """
        logger = logging.getLogger(name)

        if not logger.hasHandlers():
            if self.log_to_console:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(self.formatter)
                logger.addHandler(console_handler)

            if self.logs_file_dir is not None and self.logs_file_name is not None:
                if not os.path.exists(self.logs_file_dir):
                    os.makedirs(self.logs_file_dir)
                logs_file_path = f"{self.logs_file_dir}/{self.logs_file_name}"
                file_handler = logging.FileHandler(logs_file_path)
                file_handler.setFormatter(self.formatter)
                logger.addHandler(file_handler)

        if level is not None:
            logger.setLevel(level)
        else:
            logger.setLevel(self.log_level)

        return logger
