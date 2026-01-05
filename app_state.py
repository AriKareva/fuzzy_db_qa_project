class AppState:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.connected = False
            cls._instance.db_connection = None
            cls._instance.db_manager = None
            cls._instance.results = None
            cls._instance.cur_schema = None
            cls._instance.db_metrics = None
        return cls._instance

# Глобальный экземпляр
app_state = AppState()
