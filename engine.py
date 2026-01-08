"""
Model engine management for experiments.

Supports programmatic vLLM engine lifecycle:
- Spin up engine for a model
- Run inference
- Shutdown and swap to next model
"""

import os
import contextlib
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod


@dataclass
class ModelConfig:
    """Configuration for a model to evaluate."""
    name: str  # Display name (used in results)
    model_path: str  # HuggingFace path or local path
    
    # vLLM settings
    gpu_memory_utilization: float = 0.95
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    
    # Inference defaults
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.95
    
    # GPU selection (None = use all visible)
    cuda_visible_devices: Optional[str] = None


# Pre-configured models from models_context.md
MODEL_CONFIGS = {
    "Qwen3-8B": ModelConfig(
        name="Qwen3-8B",
        model_path="Qwen/Qwen3-8B",
        max_model_len=8192,
    ),
    "Qwen3-4B": ModelConfig(
        name="Qwen3-4B",
        model_path="Qwen/Qwen3-4B",
        max_model_len=8192,
    ),
    "Qwen3-1.7B": ModelConfig(
        name="Qwen3-1.7B",
        model_path="Qwen/Qwen3-1.7B",
        max_model_len=8192,
    ),
    "DeepSeek-R1-Distill-Qwen-7B": ModelConfig(
        name="DeepSeek-R1-Distill-Qwen-7B",
        model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        max_model_len=8192,
    ),
    "LLaMA-3.1-8B-Instruct": ModelConfig(
        name="LLaMA-3.1-8B-Instruct",
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        max_model_len=8192,
    ),
    "Ministral-8B-Instruct": ModelConfig(
        name="Ministral-8B-Instruct",
        model_path="mistralai/Ministral-8B-Instruct-2410",
        max_model_len=8192,
    ),
}


class InferenceEngine(ABC):
    """Abstract interface for inference engines."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a response for the given prompt."""
        pass
    
    @abstractmethod
    def shutdown(self):
        """Cleanup resources."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass


class VLLMEngine(InferenceEngine):
    """vLLM-based inference engine using swift.llm."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._engine = None
        self._request_config = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the vLLM engine."""
        # Set CUDA devices if specified
        if self.config.cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices
        
        try:
            from vllm import LLM, SamplingParams
            self._vllm_native = True
            
            print(f"Initializing vLLM engine for {self.config.name}...")
            print(f"  Model path: {self.config.model_path}")
            print(f"  GPU memory utilization: {self.config.gpu_memory_utilization}")
            print(f"  Tensor parallel size: {self.config.tensor_parallel_size}")
            print(f"  Max model len: {self.config.max_model_len}")
            
            self._engine = LLM(
                model=self.config.model_path,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                tensor_parallel_size=self.config.tensor_parallel_size,
                max_model_len=self.config.max_model_len,
            )
            print(f"Engine initialized successfully!")
            
        except ImportError:
            # Fallback to swift.llm if available
            try:
                from swift.llm import VllmEngine as SwiftVllmEngine, RequestConfig
                self._vllm_native = False
                
                print(f"Initializing Swift vLLM engine for {self.config.name}...")
                self._engine = SwiftVllmEngine(
                    self.config.model_path,
                    gpu_memory_utilization=self.config.gpu_memory_utilization,
                )
                self._request_config_class = RequestConfig
                print(f"Engine initialized successfully!")
                
            except ImportError:
                raise ImportError(
                    "Neither vllm nor swift.llm found. "
                    "Install with: pip install vllm  OR  pip install ms-swift"
                )
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a response."""
        try:
            if self._vllm_native:
                return self._generate_native(prompt, temperature, max_tokens, system_prompt)
            else:
                return self._generate_swift(prompt, temperature, max_tokens, system_prompt)
        except Exception as e:
            print(f"Error during generation: {e}")
            return None
    
    def _generate_native(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> Optional[str]:
        """Generate using native vLLM."""
        from vllm import SamplingParams
        
        # Build chat messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Format as chat (model-specific, using simple format)
        # Most models expect this format
        formatted_prompt = self._format_chat(messages)
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=self.config.top_p,
        )
        
        outputs = self._engine.generate([formatted_prompt], sampling_params)
        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text
        return None
    
    def _format_chat(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for chat models."""
        # Simple format that works for most instruction-tuned models
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)
    
    def _generate_swift(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> Optional[str]:
        """Generate using Swift vLLM."""
        from swift.llm import InferRequest
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        request_config = self._request_config_class(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=self.config.top_p,
            stream=False,
        )
        
        infer_request = InferRequest(messages=messages)
        responses = self._engine.infer(
            infer_requests=[infer_request],
            request_config=request_config
        )
        
        if responses and hasattr(responses[0], 'choices') and responses[0].choices:
            choice = responses[0].choices[0]
            if hasattr(choice, 'message') and choice.message:
                return getattr(choice.message, 'content', '')
        return None
    
    def shutdown(self):
        """Shutdown the engine."""
        print(f"Shutting down engine for {self.config.name}...")
        with contextlib.suppress(Exception):
            if self._engine is not None:
                if hasattr(self._engine, 'shutdown'):
                    self._engine.shutdown()
                del self._engine
                self._engine = None
        
        # Force CUDA cleanup
        with contextlib.suppress(Exception):
            import torch
            torch.cuda.empty_cache()
        
        import gc
        gc.collect()
        print("Engine shutdown complete.")
    
    @property
    def model_name(self) -> str:
        return self.config.name


class HTTPEngine(InferenceEngine):
    """HTTP-based engine for external vLLM servers (legacy compatibility)."""
    
    def __init__(self, api_url: str, model_name: str):
        self.api_url = api_url
        self._model_name = model_name
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Generate via HTTP API."""
        import requests
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=data,
                timeout=120
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"HTTP request failed: {e}")
            return None
    
    def shutdown(self):
        """No-op for HTTP engine."""
        pass
    
    @property
    def model_name(self) -> str:
        return self._model_name


@contextlib.contextmanager
def engine_context(config: ModelConfig):
    """
    Context manager for engine lifecycle.
    
    Usage:
        with engine_context(model_config) as engine:
            response = engine.generate("Hello!")
    """
    engine = VLLMEngine(config)
    try:
        yield engine
    finally:
        engine.shutdown()


def get_model_config(name: str) -> ModelConfig:
    """Get a pre-configured model by name."""
    if name in MODEL_CONFIGS:
        return MODEL_CONFIGS[name]
    
    # Assume it's a HuggingFace path
    return ModelConfig(name=name, model_path=name)

