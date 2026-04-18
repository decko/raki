from raki.adapters.protocol import SessionAdapter


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, SessionAdapter] = {}

    def register(self, adapter: SessionAdapter) -> None:
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> SessionAdapter | None:
        return self._adapters.get(name)

    def list_all(self) -> list[SessionAdapter]:
        return list(self._adapters.values())
