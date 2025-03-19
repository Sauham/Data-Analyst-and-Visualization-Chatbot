# app.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pytesseract
import pdfplumber
from docx import Document
from transformers import pipeline
import io
import re
import streamlit as st

# Load a text generation model from Hugging Face
generator = pipeline("text-generation", model="gpt2")

# Define functions to handle different file types
def read_txt(file):
    return file.read().decode("utf-8")

def read_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def read_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)

def read_csv(file):
    return pd.read_csv(file)

def read_excel(file):
    return pd.read_excel(file)

def read_file(file, file_name):
    if file_name.endswith('.txt'):
        return read_txt(file)
    elif file_name.endswith('.docx'):
        return read_docx(file)
    elif file_name.endswith('.pdf'):
        return read_pdf(file)
    elif file_name.endswith('.csv'):
        return read_csv(file)
    elif file_name.endswith('.xlsx'):
        return read_excel(file)
    elif file_name.endswith(('.png', '.jpg', '.jpeg')):
        return read_image(file)
    else:
        raise ValueError("Unsupported file type")

# Define function to answer questions using Hugging Face Transformers
def ask_question(text, question):
    # Truncate the input text to fit within the model's context size
    max_input_length = 500  # Adjust this value based on your needs
    truncated_text = text[:max_input_length]
    
    prompt = f"{truncated_text}\n\nQ: {question}\nA:"
    response = generator(prompt, max_new_tokens=100, num_return_sequences=1)
    return response[0]['generated_text']

# Define function to create visualizations
def create_visualization(data, plot_type, x=None, y=None):
    plt.figure(figsize=(10, 6))
    if plot_type == 'bar':
        sns.barplot(x=data[x], y=data[y])
    elif plot_type == 'line':
        sns.lineplot(x=data[x], y=data[y])
    elif plot_type == 'hist':
        sns.histplot(data[x])
    elif plot_type == 'scatter':
        sns.scatterplot(x=data[x], y=data[y])
    plt.title(f"{plot_type.capitalize()} Plot")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    st.pyplot(plt)

# Create the Data Analyst Agent
class DataAnalystAgent:
    def __init__(self):
        self.data = None
        self.text = None

    def upload_file(self, file, file_name):
        self.data = read_file(file, file_name)
        if isinstance(self.data, pd.DataFrame):
            self.text = self.data.to_string()
        else:
            self.text = self.data
        st.success("File uploaded successfully!")

    def ask(self, question):
        if self.text is None:
            return "Please upload a file first."
        
        # If the file is a DataFrame (CSV/Excel), perform data analysis
        if isinstance(self.data, pd.DataFrame):
            try:
                # Example: Answer questions about specific columns
                if "how many" in question.lower() and "sold" in question.lower():
                    # Extract the car model from the question
                    car_model = re.search(r"how many (.*?) were sold", question.lower()).group(1).strip()
                    # Filter the DataFrame for the specified car model
                    car_sales = self.data[self.data['Model'].str.lower() == car_model.lower()]
                    if not car_sales.empty:
                        return f"{car_sales['Sales_in_thousands'].sum() * 1000:.0f} units of {car_model} were sold."
                    else:
                        return f"No data found for {car_model}."
                elif "name of models" in question.lower():
                    # Extract the manufacturer from the question
                    manufacturer = re.search(r"name of models manufactured by (.*?)\?", question.lower()).group(1).strip()
                    # Filter the DataFrame for the specified manufacturer
                    models = self.data[self.data['Manufacturer'].str.lower() == manufacturer.lower()]['Model'].unique()
                    if len(models) > 0:
                        return f"The models manufactured by {manufacturer} are: {', '.join(models)}."
                    else:
                        return f"No models found for {manufacturer}."
                elif "what do we know about" in question.lower():
                    # Extract the manufacturer or model from the question
                    keyword = re.search(r"what do we know about (.*?)\?", question.lower()).group(1).strip()
                    # Filter the DataFrame for the specified keyword
                    if keyword in self.data['Manufacturer'].str.lower().values:
                        manufacturer_data = self.data[self.data['Manufacturer'].str.lower() == keyword]
                        return f"{keyword.capitalize()} manufactures the following models: {', '.join(manufacturer_data['Model'].unique())}. Their average sales are {manufacturer_data['Sales_in_thousands'].mean():.2f} thousand units."
                    elif keyword in self.data['Model'].str.lower().values:
                        model_data = self.data[self.data['Model'].str.lower() == keyword]
                        return f"The {keyword.capitalize()} is a {model_data['Vehicle_type'].values[0]} with an average price of {model_data['Price_in_thousands'].mean():.2f} thousand dollars."
                    else:
                        return f"No data found for {keyword}."
                else:
                    # Use the language model to answer generic questions
                    return ask_question(self.text, question)
            except Exception as e:
                return f"Error processing question: {e}"
        
        # For non-tabular files, use the language model to answer questions
        return ask_question(self.text, question)

    def visualize(self, plot_type, x=None, y=None):
        if isinstance(self.data, pd.DataFrame):
            if x is None or y is None:
                st.warning("Please specify the columns for the x and y axes.")
            else:
                create_visualization(self.data, plot_type, x, y)
        else:
            st.warning("Visualization is only supported for tabular data.")

# Streamlit App
def main():
    st.title("Data Analyst Agent")
    st.write("Upload a document and ask questions or visualize data.")

    # Initialize the agent
    agent = DataAnalystAgent()

    # File upload
    uploaded_file = st.file_uploader("Upload a document", type=['txt', 'docx', 'pdf', 'csv', 'xlsx', 'png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        agent.upload_file(uploaded_file, uploaded_file.name)

        # Ask a question
        question = st.text_input("Ask a question about the document:")
        if question:
            answer = agent.ask(question)
            st.write("Answer:", answer)

        # Visualize data (if the file is tabular)
        if isinstance(agent.data, pd.DataFrame):
            st.write("### Data Visualization")
            plot_type = st.selectbox("Choose a plot type", ['bar', 'line', 'hist', 'scatter'])
            columns = agent.data.columns.tolist()
            x_axis = st.selectbox("Select the x-axis column", columns)
            y_axis = st.selectbox("Select the y-axis column", columns)
            if st.button("Generate Plot"):
                agent.visualize(plot_type, x_axis, y_axis)

# Run the app
if __name__ == "__main__":
    main()