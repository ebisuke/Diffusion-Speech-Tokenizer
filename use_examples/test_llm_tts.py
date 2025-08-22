import torch

from models.tts.llm_tts.inference_llm_tts import TTSInferencePipeline


# Usage example
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create pipeline
    pipeline = TTSInferencePipeline.from_pretrained(
        tadicodec_path="./ckpt/TaDiCodec",
        llm_path="./ckpt/TaDiCodec-TTS-AR-Qwen2.5-0.5B",
        device=device,
    )

    # Inference on single sample
    audio = pipeline(
        text="但是 to those who 知道 her well, it was a 标志 of her unwavering 决心 and spirit.",
        prompt_text="In short, we embarked on a mission to make America great again, for all Americans.",
        prompt_speech_path="./use_examples/test_audio/trump_0.wav",
    )

    # Save audio
    import soundfile as sf

    sf.write("./use_examples/test_audio/lm_tts_output.wav", audio, 24000)
