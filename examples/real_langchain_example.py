"""
Пример реальной интеграции с LangChain и ChatOpenAI.

Чтобы запустить этот скрипт, вам понадобится:
1. Установить пакет: pip install langchain-openai
2. Задать переменную окружения OPENAI_API_KEY
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from agent_budget_guard import BudgetGuard

def main():
    # Проверяем наличие ключа (чтобы скрипт не упал с непонятной ошибкой)
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Пожалуйста, установите переменную окружения OPENAI_API_KEY")
        return

    print("🤖 Настраиваем реального агента с лимитом в $0.01...")
    
    # 1. Создаем модель
    llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=100)
    
    # 2. Создаем предохранитель (интерактивный режим)
    guard = BudgetGuard(max_budget_usd=0.01, interactive=True, verbose=True)
    
    # 3. Создаем цепочку
    prompt = ChatPromptTemplate.from_template("Расскажи мне длинную историю про {topic}")
    chain = prompt | llm

    try:
        # 4. Вызываем цепочку, передавая нашего охранника
        result = chain.invoke(
            {"topic": "роботов"},
            config={"callbacks": [guard]} # <--- Вот здесь происходит магия
        )
        print("\n✅ Ответ от LLM:", result.content)
        
    except Exception as e:
        print(f"\n⛔ Выполнение прервано: {e}")
        
    finally:
        # Выводим красивый отчет в любом случае
        guard.print_summary()

if __name__ == "__main__":
    main()