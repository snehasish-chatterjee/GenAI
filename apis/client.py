import requests
import streamlit as st

def get_cohere_response(input_text):
    response = requests.post("http://localhost:8000/essay/invoke", json={"input": {"topic": input_text}})
    return response.json()['output']

st.title("Cohere Essay Generator")
input_text = st.text_input("Enter a topic for the essay:")

if input_text:
    st.write(get_cohere_response(input_text))