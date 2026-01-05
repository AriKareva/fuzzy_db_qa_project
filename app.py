import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from app_state import app_state
from data_getter import DBConnector
from data_getter import UserDBManager
from fuzzy_system.fuzzy_logic_system import FuzzyLogicSystem

st.set_page_config(page_title="Fuzzy DB Quality Assessment", layout="wide")

st.title("Нечёткая система оценки качества БД")
st.markdown("---")

# Боковая панель для настроек подключения
with st.sidebar:
    st.header("Настройки подключения")
    
    # Поля ввода для подключения к БД
    host = st.text_input("Хост:", value='localhost')
    port = st.text_input("Порт:", value=5433)
    database = st.text_input("База данных:", value='postgres')
    username = st.text_input("Пользователь:", value='postgres')
    password = st.text_input("Пароль:", type="password")
    cur_schema = st.text_input("Схема:", value='public')
    
    connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
    
    # Кнопка подключения
    if st.button("Подключиться", type="primary", use_container_width=True):
        with st.spinner("Проверка подключения к БД..."):
            try:
                # Создаем подключение
                conn = DBConnector(host, port, username, password, database)
                conn.set_connection()
    
                app_state.connected = True
                app_state.db_connection = conn
                app_state.cur_schema = cur_schema
                st.success("**Подключение успешно установлено!**")
                st.balloons()
                
            except Exception as e:
                app_state.connected = False
                app_state.db_connection = None
                st.error(f"**Ошибка:** {str(e)}")

# Главная панель
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Процесс работы")
    st.markdown("""
    1. **Введите данные подключения** в боковую панель
    2. **Нажмите "Подключиться"** для проверки соединения
    3. **Нажмите "Оценить качество БД"** для запуска анализа
    4. **Просмотрите результаты** и графики оценки
    """)

# Показ статуса подключения
if app_state.connected:
    st.success("**Статус: Подключено к БД**")
else:
    st.warning("**Статус: Не подключено**")

st.markdown("---")

# Кнопка оценки качества (только если подключены)
if app_state.connected and st.button("Оценить качество БД", type="primary", use_container_width=True):
    with st.spinner("Запуск нечёткой оценки качества БД..."):
        try:
            db_manager = UserDBManager(db_connection=app_state.db_connection, schema=app_state.cur_schema)
            app_state.db_manager = db_manager
            
            db_metrics = db_manager.get_all_metrics()
            app_state.db_metrics = db_metrics
            print("Получены метрики БД:", db_metrics)
            
            fuzzy_logic_system = FuzzyLogicSystem()
            results = fuzzy_logic_system.evaluate_quality(db_metrics)
            app_state.results = results
            
            st.success("**Оценка завершена!**")
            
        except Exception as e:
            st.error(f"**Ошибка оценки:** {str(e)}")


if app_state.results is not None:
    st.header("Результаты оценки")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Fuzzy Score", f"{app_state.results['fuzzy_out']}")
    
    with col2:
        st.metric("Target Score", f"{app_state.results['target']}")
    
    # st.subheader("Детальные метрики БД")
    # if app_state.db_metrics is not None:
    #     st.dataframe(app_state.db_metrics.head(), use_container_width=True)

    if st.button("Очистить результаты", type="secondary"):
        app_state.results = None
        app_state.db_metrics = None
        st.rerun()

st.markdown("---")

with st.expander("Инструкция по использованию"):
    st.markdown("""
    ### Пошаговая инструкция:
    1. **Заполните параметры БД** в левой панели
    2. **Нажмите "Подключиться"** - проверьте статус
    3. **Нажмите "Оценить качество БД"** - дождитесь результатов
    4. **Изучите метрики и графики** оценки
    5. **Скачайте отчёт** для документации
    
    ### Поддерживаемые БД:
    - PostgreSQL
    
    **Все объекты сохраняются в глобальном состоянии AppState.**
    """)
