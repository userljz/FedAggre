from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


@dataclass
class FitRes:
    """Fit return for a client."""
    parameters: Any
    num_examples: Any
    metrics: Any

@dataclass
class FitIns:
    parameters: Any
    config: Dict
