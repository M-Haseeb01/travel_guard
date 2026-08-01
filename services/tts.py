from gradio_client import Client as GradioClient


def synthesize(text: str) -> bytes:

    try:

        tts_client = GradioClient(
            "MHaseeb-01/TTS"
        )

        audio_path = tts_client.predict(
            text=text,
            speed=1.3,
            api_name="/synthesize_speech"
        )

        with open(audio_path, "rb") as f:
            return f.read()

    except Exception as e:

        print("GRADIO TTS ERROR:", repr(e))

        raise
