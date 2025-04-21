# Muyan-TTS

## Highlight

Muyan-TTS is a trainable TTS model designed for podcast applications within a $50,000 budget, which is pretrained on over 100,000 hours of podcast audio data, enabling zero-shot TTS synthesis with high-quality voice generation. Furthermore, Muyan-TTS supports speaker adaptation with dozens of minutes of target speech, making it highly customizable for individual voices. Muyan-TTS makes the following four key contributions:
- **Open-sourcing two TTS models**: (i) a base model pre-trained on diverse podcast datasets, enabling zero-shot TTS synthesis, and (ii) a supervised fine-tuned (SFT) model trained on an individual speaker to enhance TTS performance.
- **Providing a detailed training methodology**: outlines the end-to-end training procedure, from the base model to speaker-specific adaptation, and release the full training code for public use.
- **Introducing a data processing pipeline**: proposes a comprehensive workflow for data collection, preprocessing, and formatting tailored to TTS model training, improving efficiency and reproducibility.
- **Optimizing inference efficiency**: develops an accelerated TTS inference framework, particularly optimizing the LLM component for faster and more efficient speech generation.

## Architecture
Muyan-TTS