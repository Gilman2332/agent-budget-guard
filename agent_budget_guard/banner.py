import pyfiglet
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

def show_banner():
    """Выводит стильный ASCII-баннер при запуске."""
    
    # Генерируем большой текст
    ascii_art = pyfiglet.figlet_format("Budget Guard", font="slant")
    
    # Раскрашиваем его в градиент (от синего к голубому)
    banner_text = Text(ascii_art, style="bold cyan")
    
    # Дописываем версию и слоган
    subtitle = Text("\n🛡️  Keep your AI agents on a budget.", style="italic magenta")
    
    # Объединяем текст
    full_text = Text.assemble(banner_text, subtitle)
    
    # Выводим в красивой рамке
    console.print(Panel(
        full_text,
        border_style="blue",
        padding=(1, 4),
        title="[bold white]v0.1.0[/bold white]",
        subtitle="[dim]Terminal Dashboard[/dim]"
    ))