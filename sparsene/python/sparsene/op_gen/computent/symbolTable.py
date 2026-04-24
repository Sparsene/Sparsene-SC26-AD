class SymbolTable:
    _instances: dict[str, "SymbolTable"] = {}

    def __init__(self):
        self._table = {}

    @classmethod
    def global_table(cls, name: str = "default") -> "SymbolTable":
        """
                    ，        
          :
            SymbolTable.global_table("arraydef")
            SymbolTable.global_table("axis")
        """
        if name not in cls._instances:
            cls._instances[name] = cls()
        return cls._instances[name]

    @classmethod
    def reset(cls):
        """         """
        cls._instances.clear()

    def clear(self):
        """       """
        self._table.clear()

    def add(self, name: str, value):
        """      ，        """
        self._table[name] = value

    def get(self, name: str):
        """    ，     value，      None"""
        return self._table.get(name, None)

    def exists(self, name: str) -> bool:
        """        """
        return name in self._table

    def remove(self, name: str):
        """    ，      """
        self._table.pop(name, None)

    def __getitem__(self, name: str):
        return self._table[name]

    def __setitem__(self, name: str, value):
        self._table[name] = value

    def __contains__(self, name: str) -> bool:
        return name in self._table

    def __str__(self):
        return str(self._table)