import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.set_page_config(layout='wide')
st.title("Math Problem Solver using Streamlit and LLM (Google Gemini)")

st.write("""
This project allows you to draw a math equation on a digital whiteboard and send it to the LLM Gemini to receive a solution.

 How to use this app:
1. Use your mouse or touchpad to draw the math equation on the whiteboard below.
2. Click the 'Solve' button to send the input to Gemini.
3. Click 'Clear Canvas' to erase everything and start over.
""")

# Configure Gemini AI
genai.configure(api_key="AIzaSyCoehrVswJRws-SnLtkRwrXhiUn-6-T1KI")
model = genai.GenerativeModel('gemini-1.5-pro')

# Create two columns
col1, col2 = st.columns([2, 1])

# Drawing canvas
with col1:
    # Specify canvas parameters in application
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    
    
    
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=400,
        width=600,
        drawing_mode="freedraw",
        key="canvas",
    )
    # Text input for problem description
    problem_text = st.text_area("Describe your problem (optional):")
    
    # Buttons for solving and clearing
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        if st.button('Solve'):
            if canvas_result.image_data is not None:
                # Convert the image data to a format Gemini can use
                img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                img = img.convert('RGB')
                
                # Include the problem description in the request
                if problem_text:
                    prompt = f"Solve this math problem: {problem_text}"
                else:
                    prompt = "Solve this math problem"
                
                response = model.generate_content([prompt, img])
                st.session_state['solution'] = response.text

# Display the solution
with col2:
    st.title("Solution")
    solution_text = st.empty()
    if 'solution' in st.session_state:
        solution_text.write(st.session_state['solution'])