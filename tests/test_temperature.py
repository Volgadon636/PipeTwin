"""Include the versioned Stage 02 tests in the standard unittest discovery."""
from experiments.temperature_v02_02.test_temperature import (  # noqa: F401
    TemperatureIntegrationTests,
    TemperatureTests,
)
