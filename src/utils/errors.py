from src.utils.config import DIC_CITY_SPECS

class InvalidCityError(ValueError):
    """Exception raised when an invalid city is provided."""
    def __init__(self, city):
        super().__init__(f"Unsupported city '{city}'. Available cities: {', '.join(DIC_CITY_SPECS.keys())}")