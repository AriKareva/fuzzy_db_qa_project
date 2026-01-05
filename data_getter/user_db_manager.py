from typing import Dict
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from data_getter.db_connector import DBConnector


class UserDBManager:
    def __init__(self, 
                 db_connection: DBConnector,
                 schema: str = "public",
                 config = None):
        if not isinstance(db_connection, DBConnector):
            raise ValueError("db_connection должен быть объектом DBConnector")
        
        
        self.db_connection: DBConnector = db_connection
        self.engine: Engine = db_connection.get_engine()
        self.schema: str = schema
        self.inspector = inspect(self.engine)
        self.view_counter = 0
        
        self.config = {
            'max_depth': 10,
            'batch_size': 1000,
        }
        if config:
            self.config.update(config)

    def get_composite_keys_ratio(self) -> float:
        table_names = self.inspector.get_table_names(schema=self.schema)
        if not table_names:
            return 0.0
            
        composite_pk_count = 0
        tables_with_pk_count = 0
        
        for table in table_names:
            pk_constraint = self.inspector.get_pk_constraint(table, schema=self.schema)
            if pk_constraint and pk_constraint['constrained_columns']:
                tables_with_pk_count += 1
                if len(pk_constraint['constrained_columns']) > 1:
                    composite_pk_count += 1
                    
        return composite_pk_count / tables_with_pk_count if tables_with_pk_count > 0 else 0.0

    def get_non_integer_keys_ratio(self) -> float:
        table_names = self.inspector.get_table_names(schema=self.schema)
        if not table_names:
            return 0.0
            
        non_int_pk_count = 0
        tables_with_pk_count = 0

        integer_types = ('INTEGER', 'BIGINT', 'SMALLINT', 'SERIAL', 'BIGSERIAL')

        for table in table_names:
            pk_constraint = self.inspector.get_pk_constraint(table, schema=self.schema)
            cols = pk_constraint.get('constrained_columns', [])
            
            if cols:
                tables_with_pk_count += 1
                is_non_int = False
                
                columns_info = self.inspector.get_columns(table, schema=self.schema)
                col_type_map = {col['name']: str(col['type']).upper() for col in columns_info}
                
                for pk_col in cols:
                    col_type = col_type_map.get(pk_col, '')
                    # Проверяем, содержит ли тип подстроку int (грубая проверка, но обычно работает для SQLALchemy типов)
                    # Более точная проверка - через isinstance(col['type'], sqlalchemy.types.Integer)
                    if not any(t in col_type for t in integer_types):
                        is_non_int = True
                        break
                
                if is_non_int:
                    non_int_pk_count += 1
                    
        return non_int_pk_count / tables_with_pk_count if tables_with_pk_count > 0 else 0.0

    def get_char_attributes_memory_waste_ratio(self) -> float:
        table_names = self.inspector.get_table_names(schema=self.schema)
        total_char_columns = 0
        wasteful_char_columns = 0
        
        for table in table_names:
            columns = self.inspector.get_columns(table, schema=self.schema)
            for col in columns:
                col_type_str = str(col['type']).upper()
                
                # Проверяем, является ли это CHAR (обычно CHAR или CHARACTER)
                # Важно: VARCHAR обычно отображается как VARCHAR
                if col_type_str.startswith('CHAR') or col_type_str.startswith('CHARACTER'):
                    # Исключаем 'CHARACTER VARYING' (это VARCHAR)
                    if 'VARYING' not in col_type_str:
                        total_char_columns += 1
                        # Получаем длину. SQLAlchemy типы хранят длину в атрибуте length
                        length = getattr(col['type'], 'length', None)
                        
                        # Если длина не указана явно, часто по стандарту 1, но лучше проверить.
                        # Если длина > 1, считаем это потенциальной растратой (лучше VARCHAR)
                        if length is not None and length > 1:
                            wasteful_char_columns += 1
                            
        return wasteful_char_columns / total_char_columns if total_char_columns > 0 else 0.0

    def get_tables_without_on_update_ratio(self) -> float:
        sql = text("""
            SELECT 
                COUNT(*) as total_fks,
                SUM(CASE WHEN update_rule IN ('NO ACTION', 'RESTRICT') THEN 1 ELSE 0 END) as no_update_action
            FROM information_schema.referential_constraints
            WHERE constraint_schema = :schema
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(sql, {"schema": self.schema}).fetchone()
            
        if not result or result.total_fks == 0:
            return 0.0
            
        return result.no_update_action / result.total_fks

    def get_tables_without_on_delete_ratio(self) -> float:
        sql = text("""
            SELECT 
                COUNT(*) as total_fks,
                SUM(CASE WHEN delete_rule IN ('NO ACTION', 'RESTRICT') THEN 1 ELSE 0 END) as no_delete_action
            FROM information_schema.referential_constraints
            WHERE constraint_schema = :schema
        """)

        with self.engine.connect() as conn:
            result = conn.execute(sql, {"schema": self.schema}).fetchone()
            
        if not result or result.total_fks == 0:
            return 0.0
            
        return result.no_delete_action / result.total_fks

    def get_all_metrics(self) -> Dict[str, float]:
        return {
            "доля_составных_ключей": self.get_composite_keys_ratio(),
            "доля_не_целочисленных_ключей": self.get_non_integer_keys_ratio(),
            "доля_char_атрибутов_с_растратой_пямяти": self.get_char_attributes_memory_waste_ratio(),
            "доля_таблиц_без_on_update": self.get_tables_without_on_update_ratio(),
            "доля_таблиц_без_on_delete": self.get_tables_without_on_delete_ratio(),
            # "доля_неатомарных_атрибутов": self.get_non_atomic_attributes_ratio() 
        }

