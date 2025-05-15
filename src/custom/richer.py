import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from rich.traceback import install

install(show_locals=True)
logging.basicConfig(level="INFO", handlers=[RichHandler()])
logger = logging.getLogger("ml-flow-tutorial")

custom_theme = Theme({"info": "bold cyan", "warning": "magenta", "danger": "bold red"})

console = Console(theme=custom_theme)
