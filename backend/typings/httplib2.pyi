from typing import Any, Dict, Optional, Tuple, Union

DEFAULT_MAX_REDIRECTS: int = 5

class Http:
    def __init__(self) -> None: ...
    def request(
        self,
        uri: str,
        method: str = "GET",
        body: Optional[Union[bytes, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        redirections: int = DEFAULT_MAX_REDIRECTS,
        connection_type: Optional[Any] = None,
        **kwargs: Any
    ) -> Tuple[Any, bytes]: ...