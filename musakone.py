from audiocraft.models import MusicGen
import streamlit as st 
import torch 
import torchaudio
import os 
import numpy as np
import base64
from deep_translator import GoogleTranslator

@st.cache_resource
def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    return model

def generate_music_tensors(description, duration: int):
    print("Description: ", description)
    print("Duration: ", duration)
    model = load_model()

    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )

    output = model.generate(
        descriptions=[description],
        progress=True,
        return_tokens=True
    )

    return output[0]


def save_audio(samples: torch.Tensor):
    """Renders an audio player for the given audio samples and saves them to a local directory.

    Args:
        samples (torch.Tensor): a Tensor of decoded audio samples
            with shapes [B, C, T] or [C, T]
        sample_rate (int): sample rate audio should be displayed with.
        save_path (str): path to the directory where audio should be saved.
    """

    print("Samples (inside function): ", samples)
    sample_rate = 32000
    save_path = "audio_output/"
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
    assert samples.dim() == 2 or samples.dim() == 3

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")
        torchaudio.save(audio_path, audio, sample_rate)

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Lataa {file_label}</a>'
    return href

st.set_page_config(
    page_icon= ":cd:",
    page_title= "Musakone"
)

def main():

    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://thewhiskeyreserve.com/cdn/shop/files/VMG_WR_LQ_511_ad690031-8e9e-45d0-ac9d-be7cf8a8e6c9.jpg?v=1700605378&width=1800');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Paikallinen musageneraattori, perustuu musicgen-small:iin ")
    time_slider = st.slider("Kuinka pitkä tehään (sekunneissa)?", 0, 180, 60)
    text_area = st.text_area("Kuvaile vapaasti mitä haluat")

    if text_area and time_slider:
        
        # Käännä suomenkielinen kuvaus englanniksi
        translator = GoogleTranslator(source='fi', target='en')
        translation = translator.translate(text_area)  # Käännettävä teksti annetaan tässä
        
        # Simuloitu JSON-näyttö
        st.json({
            'Kuvauksesi': text_area,
            'Valittu aika sekunneissa': time_slider,
            'Käännetty prompt': translation
        })

        
        # Generoidaan musiikkitensorit käyttäen käännettyä tekstiä
        st.subheader("Generoitu kappale")
        music_tensors = generate_music_tensors(translation, time_slider)

        save_music_file = save_audio(music_tensors)
        audio_filepath = 'audio_output/audio_0.wav'
        audio_file = open(audio_filepath, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)
        st.markdown(get_binary_file_downloader_html(audio_filepath, 'Audio'), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    