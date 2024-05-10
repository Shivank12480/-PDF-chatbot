import os
import uuid
import streamlit as st
import json
import PyPDF2
import numpy as np
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity

# Set OpenAI API key
openai.api_key = 'sk-proj-57HbivhT61VNsxhsUjDPT3BlbkFJK3b8CTOxSCkFOcJYJOrE'

def main():
    st.title("D-DOT-PY Pdf+ChatGPT ")

    uploaded_file = st.file_uploader("Choose a PDF file to upload", type="pdf")
    if uploaded_file is not None:
        if st.button("Read PDF"):
            save_uploaded_file(uploaded_file)
            st.write("Please wait while we learn the PDF.")
            learn_pdf(uploaded_file.name)
            st.write("PDF reading completed! Now you may ask a question")
            os.remove(uploaded_file.name)

    user_input = st.text_input("Enter your Query:")
    if st.button("Send"):
        st.write("You:", user_input)
        response = Answer_from_documents(user_input)
        st.write("Bot: "+response)

def learn_pdf(file_path):
    content_chunks = []
    pdf_file = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        content = page.extract_text()
        obj = {
            "id": str(uuid.uuid4()),
            "text": content,
            "embedding": get_embedding(content, engine='text-embedding-ada-002')
        }
        content_chunks.append(obj)

    # Save the learned data into the knowledge base...
    json_file_path = 'my_knowledgebase.json'
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = []

    for i in content_chunks:
        data.append(i)

    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    pdf_file.close()

def Answer_from_documents(user_query):
    user_query_vector = get_embedding(user_query, engine='text-embedding-ada-002')
    with open('my_knowledgebase.json', 'r', encoding="utf-8") as jsonfile:
        data = json.load(jsonfile)
        for item in data:
            item['embeddings'] = np.array(item['embedding'])

        for item in data:
            item['similarities'] = cosine_similarity(item['embedding'], user_query_vector)
        sorted_data = sorted(data, key=lambda x: x['similarities'], reverse=True)
        
        context = ''
        for item in sorted_data[:2]:
            context += item['text']

        myMessages = [
            {"role": "system", "content": "You're a helpful Assistant."},
            {"role": "user", "content": "The following is a Context:\n{}\n\n Answer the following user query according to the above given context.\n\nquery: {}".format(context, user_query)}
        ]
        
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=myMessages,
            max_tokens=200,
        )

    return response['choices'][0]['message']['content']

def save_uploaded_file(uploaded_file):
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

if __name__ == "__main__":
    main()
