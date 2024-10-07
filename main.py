import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st

# Configure Gemini AI
genai.configure(api_key="AIzaSyCoehrVswJRws-SnLtkRwrXhiUn-6-T1KI")
model = genai.GenerativeModel('gemini-1.5-pro')

st.set_page_config(layout='wide')
st.title("Math Problem Solver using Streamlit and LLM (Google Gemini)")

st.write("""
This project allows you to upload an image of a math equation and send it to the LLM Gemini to receive a solution.
How to use this app:
1. Upload an image containing the math problem you want to solve.
2. (Optional) Provide a text description of the problem.
3. Click the 'Solve' button to send the input to Gemini.
""")

# Create two columns
col1, col2 = st.columns([2, 1])

# Image upload
with col1:
    uploaded_file = st.file_uploader("Choose an image of a math problem", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

    # Text input for problem description
    problem_text = st.text_area("Describe your problem (optional):")

    # Button for solving
    if st.button('Solve') and uploaded_file is not None:
        # Include the problem description in the request
        if problem_text:
            prompt = f"Convert the math problem shown in the image into latex with the following instructions:{problem_text}"
        else:
            prompt = f"Convert the math problem shown in the image into latex with the following instructions."
        try:
            response = model.generate_content([prompt, image])
            st.session_state['solution'] = response.text
        except Exception as e:
            print('error',)
        

# Display the solution
with col2:
    st.title("Solution")
    solution_text = st.empty()
    if 'solution' in st.session_state:
        solution_text.write(st.session_state['solution'])