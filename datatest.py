import pandas as pd
import sqlite3
import streamlit as st

from langchain.chains.llm import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate


# Подключаемся к базе данных SQLite
connection = sqlite3.connect("C:\\Users\\lyuto\\OneDrive\\Рабочий стол\\UniversityProjeck\\Russian_cuisine.db")

# Запрашиваем данные из нужной таблицы
query = "SELECT * FROM 'Russian cuisine'"

# Загружаем данные в DataFrame
df = pd.read_sql_query(query, connection)

# Закрываем подключение к базе данных
connection.close()

df.head()

loader = DataFrameLoader(df, page_content_column='recipe_name')
documents = loader.load()

embeddings = OpenAIEmbeddings(openai_api_key="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")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
# создаем хранилище
db = FAISS.from_documents(texts, embeddings)
db.as_retriever()

prompt_template = """
Используя данные из рецепта, включающие алгоритм: {algorithm} и ингредиенты: {ingredients}, опишите подробно, как приготовить блюдо, чтобы читатель мог легко следовать вашему описанию.
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["algorithm", "ingredients"]
)

max_tokens = 1500

# Инициализация модели OpenAI
llm = OpenAI(temperature=0, max_tokens=max_tokens, openai_api_key="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")

# Создание LLMChain с prompt и llm
chain = LLMChain(prompt=prompt, llm=llm)

def clean_url(url):
    url = "https:" + url
    return url.replace('"', '')
def generate_recipe_description(query):
    relevants = db.similarity_search(query)
    if not relevants:
        return "Не удалось найти информацию о данном рецепте."

    # Извлечение метаданных (алгоритм и ингредиенты) из документа
    recipe_data = relevants[0].metadata
    algorithm = recipe_data.get('algorithm', '')
    ingredients = recipe_data.get('ingredients', '')
    photo_url = recipe_data.get('photo', None)

    # Генерация подробного описания рецепта с использованием LLMChain
    response = chain.run(algorithm=algorithm, ingredients=ingredients)
    return response, photo_url

# Начало Streamlit-приложения
st.title("Ассистент по поиску рецептов ChefMate")


query = st.text_input("Введите ваш запрос:")

if st.button("Получить рецепт"):
    response, photo_url = generate_recipe_description(query)
    avatar_url = "https://i.yapx.ru/XZyvC.png"  # Укажите URL или путь к аватарке ассистента
    st.image(avatar_url, width=50)
    st.write("ChefMate:")
    st.write(response)
    if photo_url:
        clean_url = clean_url(photo_url)
        st.image(clean_url, width=400)
