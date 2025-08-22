import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.tts.tadicodec.inference_tadicodec import TaDiCodecPipline
from models.tts.llm_tts.chat_template import gen_chat_prompt_for_tts


class TTSInferencePipeline(nn.Module):
    """
    TTS inference pipeline that integrates TaDiCodec and LLM models
    Uses standard LLM for autoregressive generation
    """

    def __init__(
        self,
        tadicodec_path: str,
        llm_path: str,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.llm_path = llm_path

        # Load TaDiCodec pipeline
        self.tadicodec = TaDiCodecPipline.from_pretrained(
            ckpt_dir=tadicodec_path, device=device
        )

        # Load LLM directly from pretrained
        # Try to use flash attention 2, fallback to default if not available
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_path,
                device_map=device,
                torch_dtype="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
        except Exception as e:
            print(f"Flash attention 2 not available, using default attention: {e}")
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_path,
                device_map=device,
                torch_dtype="auto",
                trust_remote_code=True,
            )

        # Load tokenizer directly from pretrained
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            trust_remote_code=True,
        )

    def tensor_to_audio_string(self, tensor):
        """Convert tensor to audio string format"""
        if isinstance(tensor, list) and isinstance(tensor[0], list):
            values = tensor[0]
        else:
            values = tensor[0].tolist() if hasattr(tensor, "tolist") else tensor[0]

        result = "<|start_of_audio|>"
        for value in values:
            result += f"<|audio_{value}|>"
        return result

    def extract_audio_ids(self, text):
        """Extract audio IDs from string containing audio tokens"""
        import re

        pattern = r"<\|audio_(\d+)\|>"
        audio_ids = re.findall(pattern, text)
        return [int(id) for id in audio_ids]

    @classmethod
    def from_pretrained(
        cls,
        tadicodec_path: str = "./ckpt/TaDiCodec",
        llm_path: str = "./ckpt/TaDiCodec-TTS-AR-Qwen2.5-0.5B",
        device: Optional[torch.device] = None,
    ):
        """
        Create pipeline from pretrained models

        Args:
            tadicodec_path: Path to TaDiCodec model
            llm_path: Path to LLM model
            device: Device to run on

        Returns:
            TTSInferencePipeline instance
        """
        resolved_device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        return cls(
            tadicodec_path=tadicodec_path,
            llm_path=llm_path,
            device=resolved_device,
        )

    @torch.no_grad()
    def __call__(
        self,
        text: str,
        prompt_text: Optional[str] = None,
        prompt_speech_path: Optional[str] = None,
        top_k: int = 50,
        top_p: float = 0.98,
        temperature: float = 1.0,
        n_timesteps: int = 25,
        return_code: bool = False,
    ):
        """
        Perform TTS inference

        Args:
            text: Target text to synthesize
            prompt_text: Prompt text for conditioning
            prompt_speech_path: Path to prompt audio file
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            temperature: Temperature for sampling
            n_timesteps: Number of diffusion timesteps
            return_code: Whether to return audio codes instead of audio

        Returns:
            Generated audio array or audio codes
        """
        # Get prompt audio codes
        if prompt_speech_path:
            prompt_speech_code = self.tadicodec(
                speech_path=prompt_speech_path, return_code=True, text=""
            )
        else:
            raise ValueError("prompt_speech_path is required")

        # Use standard LLM for autoregressive generation
        # TODO: add a json style chat prompt template
        prompt = gen_chat_prompt_for_tts(
            (prompt_text or "") + text,
            "phi-3" if "Phi" in self.llm_path else "qwen2",
        ) + self.tensor_to_audio_string(prompt_speech_code)

        input_ids = self.tokenizer.encode(prompt)
        generate_ids = self.llm.generate(
            input_ids=torch.tensor(input_ids).unsqueeze(0).to(self.device),
            min_new_tokens=12,
            max_new_tokens=400,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

        output = self.tokenizer.decode(generate_ids[0], skip_special_tokens=False)

        combine_speech_code = self.extract_audio_ids(output)
        indices = torch.tensor(combine_speech_code).unsqueeze(0).long().to(self.device)

        if return_code:
            return indices

        # Decode audio
        text_token_ids = self.tadicodec.tokenize_text(text, prompt_text)
        prompt_mel = self.tadicodec.extract_mel_feature(prompt_speech_path)

        rec_mel = self.tadicodec.decode(
            indices=indices,
            text_token_ids=text_token_ids,
            prompt_mel=prompt_mel,
            n_timesteps=n_timesteps,
        )

        rec_audio = (
            self.tadicodec.vocoder_model(rec_mel.transpose(1, 2))
            .detach()
            .cpu()
            .numpy()[0][0]
        )

        return rec_audio
