from transformers import pipeline
import ollama
import pyttsx3
import os
import streamlit as st

def img2txt(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    text = image_to_text(url, max_new_tokens=200)[0]["generated_text"]
    return text

def generate_story(text_generated):
    """Generates a video script based on the provided prompt.

    Args:
        prompt: The user-provided prompt for the script.

    Returns:
        The generated video script as a string.
    """

    system_prompt = (
        "You are a helpful assistant that generates detailed and creative stories based on user prompts. "
    )

    response = ollama.chat(model='mistral', messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': text_generated},
    ])

    return response['message']['content']

def txt2speech(message,filename = 'output.wav'):
    engine = pyttsx3.init()  # Initialize pyttsx3 engine

    # Optional: Set speech rate (default is 150 words per minute)
    engine.setProperty('rate', 175)  # Adjust voice rate as needed

    # Speak the generated story
    engine.save_to_file(message,filename)
    engine.runAndWait()  # Wait for speech to finish



st.set_page_config(page_title="Image to Audio Story",page_icon="🔊")
st.header("Turn an Image into an Audio Story")
uploaded_file = st.file_uploader("Chooose an Image...",type="jpg")

if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name,"wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file,caption="Uploaded Image.",use_column_width=True)
        scenario = img2txt(uploaded_file.name)
        story = generate_story(scenario)
        txt2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        st.audio("output.wav")


