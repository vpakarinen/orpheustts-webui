import gradio as gr
import logging
import wave
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import torch
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        logger.info("CUDA is available! Using GPU acceleration.")
    else:
        logger.warning("CUDA is not available. Orpheus TTS requires CUDA for optimal performance.")
except Exception as e:
    has_cuda = False
    logger.warning(f"Error checking CUDA availability: {e}")
    logger.warning("Orpheus TTS requires CUDA for optimal performance.")

model = None

VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
EMOTIVE_TAGS = ["<laugh>", "<chuckle>", "<sigh>", "<cough>", "<sniffle>", "<groan>", "<yawn>", "<gasp>"]

def load_model():
    """Load the Orpheus TTS model"""
    global model
    if model is None:
        try:
            if not has_cuda:
                logger.error("CUDA not available. Orpheus TTS requires a CUDA-capable GPU.")
                return "Error: CUDA not available. Orpheus TTS requires a CUDA-capable GPU."
                
            logger.info("Starting to load Orpheus model...")
            
            from orpheus_tts.engine_class import OrpheusModel as OriginalOrpheusModel
            from orpheus_tts import OrpheusModel

            original_setup = OriginalOrpheusModel._setup_engine
            
            def patched_setup_engine(self):
                from vllm.engine.arg_utils import AsyncEngineArgs
                
                engine_args = AsyncEngineArgs(
                    model=self.model_name,
                    dtype="float16",
                    max_model_len=16384,
                    gpu_memory_utilization=0.95,
                    trust_remote_code=True
                )
                
                from vllm.engine.async_llm_engine import AsyncLLMEngine
                return AsyncLLMEngine.from_engine_args(engine_args)
            
            OriginalOrpheusModel._setup_engine = patched_setup_engine
            model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")
            OriginalOrpheusModel._setup_engine = original_setup
            
            logger.info("Orpheus model loaded successfully!")
            return "Model loaded successfully!"
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error loading model: {str(e)}"
    return "Model already loaded!"

def generate_speech(text, voice, emotive_tag=None, temperature=1.0, repetition_penalty=1.1):
    """Generate speech using Orpheus TTS"""
    global model
    
    logger.info(f"Generate speech request: voice={voice}, text={text[:30]}...")
    
    if not has_cuda:
        logger.error("CUDA not available for speech generation")
        return None, "Error: CUDA not available. Orpheus TTS requires a CUDA-capable GPU."
    
    if model is None:
        logger.info("Model not loaded yet, attempting to load...")
        try:
            result = load_model()
            logger.info(f"Model load result: {result}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, f"Error loading model: {str(e)}"
    
    if emotive_tag and emotive_tag != "None":
        text = f"{emotive_tag} {text}"
        logger.info(f"Added emotive tag, new text: {text[:50]}...")
    
    start_time = time.monotonic()
    
    try:
        logger.info("Starting speech generation...")
        syn_tokens = model.generate_speech(
            prompt=text,
            voice=voice,
            temperature=temperature,
            repetition_penalty=repetition_penalty
        )

        logger.info("Speech generation completed successfully")
        output_file = "output.wav"
        logger.info(f"Writing to output file: {output_file}")
        
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            
            total_frames = 0
            
            for audio_chunk in syn_tokens:
                frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                total_frames += frame_count
                wf.writeframes(audio_chunk)
            
            duration = total_frames / wf.getframerate()
            logger.info(f"Generated {duration:.2f} seconds of audio")
        
        end_time = time.monotonic()
        processing_time = end_time - start_time
        logger.info(f"Total processing time: {processing_time:.2f} seconds")
        
        return (
            output_file, 
            f"Generated {duration:.2f} seconds of audio in {processing_time:.2f} seconds"
        )
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, f"Error generating speech: {str(e)}"

with gr.Blocks(title="Orpheus TTS Web UI", theme=gr.themes.Base()) as demo:
    gr.Markdown(
        """
        <h1 style='text-align: center;'>Orpheus TTS Web UI</h1>
        """
    )
    
    if not has_cuda:
        gr.Markdown(
            """
            ⚠️ **WARNING: CUDA is not available on this system** ⚠️
            
            Orpheus TTS requires a CUDA-capable GPU for operation.
            Please run this application on a system with a compatible NVIDIA GPU.
            """
        )
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Text to speak",
                placeholder="Enter the text you want to convert to speech...",
                lines=5
            )
            
            with gr.Row():
                voice_selector = gr.Dropdown(
                    choices=VOICES, 
                    value="tara", 
                    label="Voice"
                )
                
                emotive_tag = gr.Dropdown(
                    choices=["None"] + EMOTIVE_TAGS, 
                    value="None", 
                    label="Emotive Tag"
                )
            
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.1, 
                    maximum=2.0, 
                    value=1.0, 
                    step=0.1, 
                    label="Temperature",
                    info="Higher values make output more random"
                )
                
                repetition_penalty = gr.Slider(
                    minimum=1.1, 
                    maximum=2.0, 
                    value=1.1, 
                    step=0.1, 
                    label="Repetition Penalty",
                    info="Required to be ≥ 1.1 for stable generations. Higher values make the model speak faster"
                )
            
            generate_btn = gr.Button("Generate Speech", variant="primary")

            audio_output = gr.Audio(label="Generated Speech", type="filepath")
            stats_output = gr.Textbox(label="Statistics", interactive=False)
        
    def auto_load_model():
        load_model()
    
    demo.load(fn=auto_load_model)

    def generate_with_custom(text, voice, emotive_tag, temperature, repetition_penalty):
        return generate_speech(text, voice, emotive_tag, temperature, repetition_penalty)
    
    generate_btn.click(
        fn=generate_with_custom,
        inputs=[text_input, voice_selector, emotive_tag, temperature, repetition_penalty],
        outputs=[audio_output, stats_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
