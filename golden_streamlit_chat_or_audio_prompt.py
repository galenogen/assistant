import streamlit as st
from openai import OpenAI
import os
from pydub import AudioSegment
import whisper
import tempfile

# Create the OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

st.set_page_config(
    page_title="Home",
    layout="wide",
)

# Title of webpage
st.title("Galenogen Healthcare Assistant")

# Set up sidebar to configure parameters
with st.sidebar:   
    st.markdown("# Configuration")

    # Select the two models to be compared
    model = 'gpt-4o' # default model, can be overridden by selection below
    model = st.selectbox('Select an LLM model',('gpt-4o', 'perplexity', 'gpt-4o-turbo', 'gpt-4o-mini'), key="model")
    health_specialization = st.selectbox('Select a health specialzation',('Psychiatry', 'Anesthesiology', 'Cardiology', 'Dermatology', 'Emergency Medicine', 'Gastroenterology', 'Internal medicine', 'Neurology', 'Obstetrics & Gynecology', 'Oncology', 'Opthalmology', 'Orthopedics', 'Pediatrics', 'Radiology'), key="specialization")

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

whisper_model = load_whisper_model()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to process audio and get transcript
def process_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        audio_segment = AudioSegment.from_file(audio_file)
        audio_segment.export(tmp_file.name, format="wav")
        result = whisper_model.transcribe(tmp_file.name)
    return result["text"]

# Audio input
audio_input = st.audio_input("Speak your message")
if audio_input:
    transcript = process_audio(audio_input)
    st.session_state.messages.append({"role": "user", "content": transcript})
    with st.chat_message("user"):
        st.markdown(transcript)

# Text input
text_input = st.chat_input("Type your message")
if text_input:
    st.session_state.messages.append({"role": "user", "content": text_input})
    with st.chat_message("user"):
        st.markdown(text_input)

# Process the latest message with LLM (e.g., GPT-4)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat.completions.create(
            model=model,
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            stream=True,
        ):
            full_response += response.choices[0].delta.content or ""
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
