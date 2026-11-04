import threading
import warnings
import math
import copy
import tiktoken
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

from .exceptions import BudgetExceededError
from .pricing import PRICING, DEFAULT_PRICING

console = Console()

class BudgetGuard(BaseCallbackHandler):
    """LangChain callback для жесткого контроля бюджета."""
    
    raise_error: bool = True

    def __init__(
        self,
        max_budget_usd: float,
        pricing: dict | None = None,
        default_max_output_tokens: int = 1000,
        verbose: bool = False,
        interactive: bool = True,
        unknown_model_policy: str = "raise"
    ):
        if not isinstance(max_budget_usd, (int, float)) or isinstance(max_budget_usd, bool) or not math.isfinite(max_budget_usd) or max_budget_usd <= 0:
            raise ValueError("max_budget_usd должен быть положительным конечным числом")
        if not isinstance(default_max_output_tokens, int) or isinstance(default_max_output_tokens, bool) or default_max_output_tokens < 0:
            raise ValueError("default_max_output_tokens должен быть неотрицательным целым числом")
        if unknown_model_policy not in ["warn", "raise", "allow"]:
            raise ValueError("unknown_model_policy должен быть 'warn', 'raise' или 'allow'")

        self._initial_budget = max_budget_usd
        self.max_budget_usd = max_budget_usd
        self.pricing = pricing or PRICING
        self.default_max_output_tokens = default_max_output_tokens
        self.verbose = verbose
        self.interactive = interactive
        self.unknown_model_policy = unknown_model_policy
        
        self.total_spent = 0.0
        self.call_count = 0
        self.history = []
        
        self._lock = threading.Lock()
        self._reserved_cost = 0.0
        self._reservations = {} 
        self._tokenizers = {}
        
        self._validate_pricing(self.pricing)

    def _validate_pricing(self, pricing: dict):
        if not isinstance(pricing, dict):
            raise ValueError("pricing должен быть словарем")
        for model, rates in pricing.items():
            if not isinstance(rates, dict) or "input" not in rates or "output" not in rates:
                raise ValueError(f"Тариф для '{model}' должен содержать 'input' и 'output'")
            for k in ["input", "output"]:
                val = rates[k]
                if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val) or val < 0:
                    raise ValueError(f"Тариф '{k}' для '{model}' должен быть неотрицательным конечным числом")

    def _get_tokenizer(self, model_name: str):
        if model_name in self._tokenizers:
            return self._tokenizers[model_name]
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        self._tokenizers[model_name] = enc
        return enc

    def _get_model_pricing(self, model_name: str, fallback_to_default: bool = False) -> dict:
        if model_name in self.pricing:
            return self.pricing[model_name]
        
        best_match = ""
        for key in self.pricing:
            if model_name.startswith(key) and len(key) > len(best_match):
                best_match = key
                
        if best_match:
            return self.pricing[best_match]
        
        if fallback_to_default:
            return DEFAULT_PRICING
            
        if self.unknown_model_policy == "raise":
            raise ValueError(f"Неизвестная модель '{model_name}'. Нет данных о тарифах.")
        elif self.unknown_model_policy == "warn":
            warnings.warn(f"Неизвестная модель '{model_name}'. Используется DEFAULT_PRICING.")
            
        return DEFAULT_PRICING
    
    def _estimate_cost(self, model_name: str, prompts: list[str], max_tokens: int | None, n: int = 1) -> float:
        pricing = self._get_model_pricing(model_name)
        enc = self._get_tokenizer(model_name)
        
        input_tokens = sum(len(enc.encode(p)) for p in prompts)
        # P1: Для Claude возвращаем консервативный запас (1.2x)
        if "claude" in model_name:
            input_tokens = int(input_tokens * 1.2)
            
        max_out = max_tokens if max_tokens is not None else self.default_max_output_tokens
        output_tokens = max_out * len(prompts) * n
        
        return input_tokens * pricing["input"] + output_tokens * pricing["output"]

    def _extract_usage(self, response: LLMResult) -> tuple[int, int, bool]:
        total_input = 0
        total_output = 0
        has_input = False
        has_output = False
        
        # P1: Суммируем usage корректно, чтобы не дублировать при n > 1
        # В LangChain статистика обычно возвращается целиком в llm_output или usage_metadata
        for gen_list in response.generations:
            for gen in gen_list:
                if hasattr(gen, "message") and hasattr(gen.message, "usage_metadata") and gen.message.usage_metadata:
                    meta = gen.message.usage_metadata
                    if "input_tokens" in meta:
                        total_input += meta["input_tokens"]
                        has_input = True
                    if "output_tokens" in meta:
                        total_output += meta["output_tokens"]
                        has_output = True
                        
        if not has_input and not has_output and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            if "prompt_tokens" in token_usage:
                total_input = token_usage["prompt_tokens"]
                has_input = True
            elif "input_tokens" in token_usage:
                total_input = token_usage["input_tokens"]
                has_input = True
            if "completion_tokens" in token_usage:
                total_output = token_usage["completion_tokens"]
                has_output = True
            elif "output_tokens" in token_usage:
                total_output = token_usage["output_tokens"]
                has_output = True
                
        return total_input, total_output, (has_input and has_output)

    def on_llm_start(self, serialized, prompts, **kwargs):
        # P0: Поддержка ChatOpenAI (ключ 'model') и старых ключей
        params = kwargs.get("invocation_params", {})
        model_name = params.get("model_name") or params.get("model") or params.get("deployment_name")
        if not model_name:
            raise ValueError("Не удалось определить модель LLM (нет model_name, model или deployment_name)")
            
        # P1: Поддержка max_completion_tokens (новый стандарт OpenAI)
        max_tokens = params.get("max_tokens") or params.get("max_completion_tokens")
        n = params.get("n", 1)
        
        if max_tokens is not None and (not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0):
            raise ValueError("max_tokens должен быть неотрицательным целым числом")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError("n должен быть положительным целым числом")
            
        run_id = kwargs.get("run_id")
        if not run_id:
            raise ValueError("run_id отсутствует. Убедитесь, что используется LangChain Callback Manager.")

        estimated = self._estimate_cost(model_name, prompts, max_tokens, n)
        enc = self._get_tokenizer(model_name)
        input_tokens = sum(len(enc.encode(p)) for p in prompts)
        
        # P0: Сохраняем словарь тарифов в резерв
        current_pricing = dict(self._get_model_pricing(model_name))
        
        needs_approval = False
        with self._lock:
            current_committed = self.total_spent + self._reserved_cost
            if current_committed + estimated > self.max_budget_usd:
                needs_approval = True
            else:
                self._reserved_cost += estimated
                self._reservations[run_id] = {
                    "estimated_cost": estimated, 
                    "model_name": model_name, 
                    "pricing": current_pricing,
                    "input_tokens": input_tokens
                }

        if needs_approval:
            if self.interactive:
                console.print(Panel(
                    f"[bold red]⚠️  Достигнут лимит бюджета![/bold red]\n\n"
                    f"Потрачено: [yellow]${self.total_spent:.4f}[/yellow] | "
                    f"В процессе: [yellow]${self._reserved_cost:.4f}[/yellow]\n"
                    f"Лимит: [green]${self.max_budget_usd:.2f}[/green]\n"
                    f"Следующий вызов: ~[yellow]${estimated:.4f}[/yellow]",
                    title="[bold yellow]Agent Budget Guard[/bold yellow]",
                    border_style="red"
                ))
                
                choice = Prompt.ask("   [bold cyan]Продолжить (превысить лимит разово)?[/bold cyan]", choices=["y", "n"], default="n")
                
                if choice != 'y':
                    raise BudgetExceededError(self.total_spent, estimated, self.max_budget_usd, model_name)
                else:
                    console.print("   [bold green]▶️  Продолжаем (разовое превышение)...[/bold green]\n")
                    with self._lock:
                        self._reserved_cost += estimated
                        self._reservations[run_id] = {
                            "estimated_cost": estimated, "model_name": model_name, 
                            "pricing": current_pricing, "input_tokens": input_tokens
                        }
            else:
                raise BudgetExceededError(self.total_spent, estimated, self.max_budget_usd, model_name)

    def on_llm_end(self, response: LLMResult, **kwargs):
        run_id = kwargs.get("run_id")
        if not run_id:
            return
            
        with self._lock:
            res = self._reservations.get(run_id)
            if not res:
                return

            # P0: Используем правильный тариф и модель из резерва
            model_name = res["model_name"]
            pricing = res["pricing"]
            
            api_model = response.llm_output.get("model_name") if response.llm_output else None
            if api_model and api_model != model_name and self.verbose:
                console.print(f"  [dim yellow]ℹ️ API вернул модель '{api_model}', считаем по тарифу '{model_name}'[/dim yellow]")

            input_tokens, output_tokens, is_complete = self._extract_usage(response)
            
            if not is_complete:
                actual_cost = res["estimated_cost"]
                if self.verbose:
                    console.print(f"  [dim yellow]⚠️ Неполная статистика, списан резерв ${actual_cost:.4f}[/dim yellow]")
            else:
                actual_cost = round(input_tokens * pricing["input"] + output_tokens * pricing["output"], 6)
            
            self.call_count += 1
            self.total_spent = round(self.total_spent + actual_cost, 6)
            
            self.history.append({
                "model": model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": actual_cost,
                "total": self.total_spent,
                "is_estimated": not is_complete
            })
            
            # P0: Атомарно удаляем резерв ТОЛЬКО после успешного расчета
            self._reservations.pop(run_id, None)
            self._reserved_cost = max(0.0, self._reserved_cost - res["estimated_cost"])
            
            if self.verbose:
                console.print(f"  [dim]├──[/dim] [green]Call #{self.call_count}[/green]: "
                              f"[yellow]${actual_cost:.4f}[/yellow] [dim](total: ${self.total_spent:.4f}/${self.max_budget_usd:.2f})[/dim]")

    def on_llm_error(self, error: Exception, **kwargs):
        run_id = kwargs.get("run_id")
        if not run_id:
            return
            
        with self._lock:
            res = self._reservations.pop(run_id, None)
            if res:
                # P0: Запрос ушел к провайдеру и мог быть оплачен. Списываем резерв!
                actual_cost = res["estimated_cost"]
                self.total_spent = round(self.total_spent + actual_cost, 6)
                self._reserved_cost = max(0.0, self._reserved_cost - actual_cost)
                
                self.call_count += 1
                self.history.append({
                    "model": res["model_name"],
                    "input_tokens": res["input_tokens"],
                    "output_tokens": 0,
                    "cost": actual_cost,
                    "total": self.total_spent,
                    "is_estimated": True,
                    "error": str(error)
                })
                
                if self.verbose:
                    console.print(f"  [dim red]⚠️ Ошибка API. Списана резервная сумма ${actual_cost:.4f}[/dim red]")

    def summary(self) -> dict:
        with self._lock:
            return {
                "total_spent": round(self.total_spent, 4),
                "budget": round(self._initial_budget, 4),
                "remaining": round(self._initial_budget - self.total_spent - self._reserved_cost, 4),
                "reserved": round(self._reserved_cost, 4),
                "call_count": self.call_count,
                "history": copy.deepcopy(self.history) # P1: Глубокая копия
            }
        
    def print_summary(self):
        data = self.summary()
        table = Table(title="📊 Итоговый отчет Agent Budget Guard", border_style="blue")
        table.add_column("Метрика", style="cyan", no_wrap=True)
        table.add_column("Значение", style="magenta")
        
        table.add_row("Всего потрачено", f"${data['total_spent']:.4f}")
        table.add_row("Изначальный лимит", f"${data['budget']:.4f}")
        table.add_row("В процессе (резерв)", f"${data['reserved']:.4f}")
        table.add_row("Доступный остаток", f"${data['remaining']:.4f}")
        table.add_row("Количество вызовов", str(data['call_count']))
        
        console.print("\n")
        console.print(table)