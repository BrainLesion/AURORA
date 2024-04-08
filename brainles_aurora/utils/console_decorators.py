from functools import wraps
from rich.console import Console

CITATION_LINK = "https://github.com/BrainLesion/AURORA#citation"


def citation_reminder(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        console = Console()
        console.rule("Thank you for using [bold]AURORA[/bold]")
        console.print(
            f"Please support our development by citing the papers listed here:",
            justify="center",
        )
        console.print(
            f"{CITATION_LINK} -- Thank you!",
            justify="center",
        )
        console.rule()
        console.line()
        return func(*args, **kwargs)

    return wrapper
