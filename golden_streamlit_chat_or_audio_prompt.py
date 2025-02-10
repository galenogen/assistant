import streamlit as st
from streamlit_float import float_init, float_parent
from openai import OpenAI
import whisper
from pydub import AudioSegment
# from gtts import gTTS
# from elevenlabs.client import ElevenLabs
# from elevenlabs import play
import os
import tempfile
from dotenv import load_dotenv
load_dotenv()

# # Create the ElevenLabs client (not activated at present)
# el_api_key = os.getenv("ELEVENLABS_API_KEY")
# tts_client = ElevenLabs(api_key=el_api_key)

# Set the page to a wide format
st.set_page_config(
    page_title="Home",
    layout="wide",
)

# Title of webpage
st.title("Galenogen Healthcare Assistant")

# Set up sidebar to configure parameters
with st.sidebar:   
    st.markdown("# Configuration")

    # Select input method - text or voice
    input_method = st.radio(
        "Choose typed or voice input:",
        ("Typed", "Spoken"),
    )

    st.write("Output will always be displayed on screen")

    # Select the two models to be compared
    llm_vendor = st.selectbox('Select the LLM vendor',('Perplexity', 'OpenAI'), key="model")
    if llm_vendor == "Perplexity":
        model = 'sonar-pro'
    elif llm_vendor == 'OpenAI':
        model = 'gpt-4o'
    else:
        model = 'sonar-pro'

    # Create the LLM client based on the selected vendor
    if llm_vendor == "Perplexity":
        api_key = os.getenv("PERPLEXITY_API_KEY")
        client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
    elif llm_vendor == 'OpenAI':
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
    else:
        api_key = os.getenv("PERPLEXITY_API_KEY")
        client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

    # Select the health specialization
    health_specialization = st.selectbox('Select a health specialzation',('None', 'Psychiatry'), key="specialization")

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

# # Google TTS: Function to convert text to speech and play it (not activated at present)
# def speak(text):
#     tts = gTTS(text=text, lang='en')
#     with tempfile.NamedTemporaryFile(delete=True) as fp:
#         tts.save(f"{fp.name}.mp3")
#         os.system(f"afplay {fp.name}.mp3")  # Use 'afplay' for MacOS or 'start' for Windows
#         # os.system(f"start {fp.name}.mp3")  # Use 'afplay' for MacOS or 'start' for Windows

# # ElevenLabs TTS: Function to convert text to speech and play it (not activated at present)
# def speak(text):
#     audio = tts_client.text_to_speech.convert(
#         text=text,
#         voice_id="JBFqnCBsd6RMkjVDRZzb",
#         model_id="eleven_multilingual_v2",
#         output_format="mp3_44100_128",
#     )
#     play(audio)

# Perform different actions based on whether input is text or audio
user_input = None
if input_method == "Typed": # Text input
    # Get new user input from chat_input()
    user_input = st.chat_input("Type your message")
else:
    # Get new audio input
    audio_input = st.audio_input("Speak your message")
    if audio_input:
        transcript = process_audio(audio_input)
        user_input = transcript

# Process the latest message with LLM (e.g., GPT-4)
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]
    )
    content = response.choices[0].message.content

    # Add AI response to chat history
    with st.chat_message("assistant"):
        st.write(content)
        st.session_state.messages.append({"role": "assistant", "content": content})
