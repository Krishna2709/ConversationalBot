import gradio as gr
from transformers import pipeline
import numpy as np
from openai import OpenAI
import elevenlabs
import tempfile
import constants

client = OpenAI(api_key=constants.OPENAI_API_KEY)
elevenlabs.set_api_key(constants.ELEVENLABS_API_KEY)

# ASR model
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

# handling transcription
def transcribe_and_respond(audio):

    sr, data = audio # sample rate, data
    data = data.astype(np.float32) # convert to 32-bit float
    data /= np.max(np.abs(data)) # normalize

    # transcribe the audio
    transcription = transcriber({'sampling_rate': sr, 'raw': data})['text']
    print('Transcription:', transcription)


    # send transcript to GPT model and get a response
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = [
            {
                "role": "system",
                "content": "You are a highly intelligent AI system, that keeps its answer short and max 1000 characters."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )

    text_response = response.choices[0].message.content
    print('Response:', text_response)

    # convert the response to audio
    audio_response = elevenlabs.generate(text=text_response, voice="Kayla - Nurturing and Caring")

    # save the audio into a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(audio_response)
        temp_audio_path = temp_audio.name

    return temp_audio_path


demo = gr.Interface(
    fn = transcribe_and_respond,
    inputs = gr.Audio(sources=["microphone"]),
    outputs = gr.Audio(sources=["upload"], autoplay=True)  
)


if __name__ == "__main__":
    demo.launch(debug=True) 