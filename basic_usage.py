"""Пример: как BudgetGuard останавливает агента до перерасхода."""
from agent_budget_guard import BudgetGuard, BudgetExceededError
from agent_budget_guard.banner import show_banner
from langchain_core.outputs import LLMResult, ChatGeneration
from langchain_core.messages import AIMessage
from rich.console import Console

console = Console()

def simulate_agent_run():
    show_banner()
    
    console.print("\n[bold cyan]🤖 Запуск симуляции AI-агента...[/bold cyan]\n")
    
    # Бюджет 1 цент для демонстрации
    guard = BudgetGuard(max_budget_usd=0.01, verbose=True)
    
    for i in range(10):
        run_id = f"run_{i}" # Генерируем уникальный ID для каждого вызова
        
        try:
            # Имитируем вызов LangChain (передаем run_id!)
            guard.on_llm_start(
                {},
                [f"Промпт номер {i} " * 100],
                invocation_params={"model_name": "gpt-4o", "max_tokens": 500},
                run_id=run_id
            )
            
            # Имитируем ответ от нейросети (передаем run_id!)
            message = AIMessage(
                content="Ответ",
                usage_metadata={"input_tokens": 200, "output_tokens": 100, "total_tokens": 300}
            )
            fake_response = LLMResult(
                generations=[[ChatGeneration(message=message)]],
                llm_output={"model_name": "gpt-4o"}
            )
            
            guard.on_llm_end(fake_response, run_id=run_id)
            
            console.print(f"  [dim]├──[/dim] [green]Вызов {i}: разрешён и выполнен[/green]")
            
        except BudgetExceededError:
            # Ошибку мы уже красиво вывели внутри guard.py, здесь просто прерываем цикл
            break
    
    # Выводим итоговую таблицу
    guard.print_summary()

if __name__ == "__main__":
    simulate_agent_run()