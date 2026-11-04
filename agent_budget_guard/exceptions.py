class BudgetExceededError(Exception):
    """Выбрасывается, когда следующая операция превысит бюджет."""
    
    def __init__(self, spent: float, estimated: float, limit: float, model: str):
        self.spent = spent
        self.estimated = estimated
        self.limit = limit
        self.model = model
        super().__init__(
            f"Budget exceeded for model '{model}'. "
            f"Already spent: ${spent:.4f}, "
            f"next call would cost ~${estimated:.4f}, "
            f"limit: ${limit:.2f}. "
            f"Agent stopped before making the call."
        )