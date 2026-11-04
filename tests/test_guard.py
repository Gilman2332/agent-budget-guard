import pytest
import math
from langchain_core.outputs import LLMResult, Generation, ChatGeneration
from langchain_core.messages import AIMessage
from agent_budget_guard import BudgetGuard, BudgetExceededError

def make_fake_response(model="gpt-4o", prompt_tokens=100, completion_tokens=50):
    message = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": prompt_tokens, "output_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}
    )
    gen = ChatGeneration(message=message)
    return LLMResult(generations=[[gen]], llm_output={"model_name": model})

def test_validation_nan():
    with pytest.raises(ValueError):
        BudgetGuard(max_budget_usd=float('nan'))
    with pytest.raises(ValueError):
        BudgetGuard(max_budget_usd=1.0, default_max_output_tokens=float('nan'))

def test_model_key_extraction():
    """P0: Проверка, что ChatOpenAI передает 'model', а не 'model_name'."""
    guard = BudgetGuard(max_budget_usd=1.0)
    # Симулируем вызов как от ChatOpenAI
    guard.on_llm_start({}, ["hello"], invocation_params={"model": "gpt-4o"}, run_id="test_model")
    assert guard._reservations["test_model"]["model_name"] == "gpt-4o"

def test_unknown_model_fallback():
    """P0: Если API вернул неизвестное имя, мы должны посчитать по тарифу из резерва."""
    guard = BudgetGuard(max_budget_usd=1.0, unknown_model_policy="allow")
    # Запускаем с gpt-4o
    guard.on_llm_start({}, ["hello"], invocation_params={"model_name": "gpt-4o"}, run_id="test2")
    
    # Ответ пришел с неизвестной моделью
    response = make_fake_response(model="unknown-super-model")
    guard.on_llm_end(response, run_id="test2")
    
    # Модель в истории должна остаться gpt-4o
    assert guard.history[0]["model"] == "gpt-4o"

def test_atomicity_and_partial_stats():
    """P0: Проверка списания резерва при неполной статистике (отсутствует ключ)."""
    guard = BudgetGuard(max_budget_usd=1.0)
    guard.on_llm_start({}, ["hello"], invocation_params={"model_name": "gpt-4o"}, run_id="test1")
    
    # Создаем ответ, где ВООБЩЕ НЕТ output_tokens
    message = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 500, "total_tokens": 500} # Нет output_tokens!
    )
    gen = ChatGeneration(message=message)
    response = LLMResult(generations=[[gen]], llm_output={"model_name": "gpt-4o"})
    
    guard.on_llm_end(response, run_id="test1")
    
    assert guard.history[0]["is_estimated"] == True
    assert guard.total_spent > 0

def test_api_error_charges_reserved():
    """P0: Если API упал, мы не делаем запрос бесплатным."""
    guard = BudgetGuard(max_budget_usd=1.0)
    guard.on_llm_start({}, ["hello"], invocation_params={"model_name": "gpt-4o"}, run_id="test_err")
    
    # Симулируем ошибку
    guard.on_llm_error(Exception("Timeout"), run_id="test_err")
    
    # Резерв должен быть списан в расходы!
    assert guard.total_spent > 0
    assert guard.summary()["reserved"] == 0.0

def test_budget_exceeded_raises():
    guard = BudgetGuard(max_budget_usd=0.0001, interactive=False) 
    with pytest.raises(BudgetExceededError):
        guard.on_llm_start(
            {}, ["a very long prompt " * 1000],
            invocation_params={"model_name": "gpt-4o", "max_tokens": 5000},
            run_id="test3"
        )