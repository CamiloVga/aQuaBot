import spaces
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch
import logging
import sys
import os
from accelerate import infer_auto_device_map, init_empty_weights

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get HuggingFace token from environment variable
hf_token = os.environ.get('HUGGINGFACE_TOKEN')
if not hf_token:
    logger.error("HUGGINGFACE_TOKEN environment variable not set")
    raise ValueError("Please set the HUGGINGFACE_TOKEN environment variable")

# Define the model name
model_name = "meta-llama/Llama-2-7b-chat-hf"

try:
    logger.info("Starting model initialization...")
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token
    )
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded successfully")

    # Load model with basic configuration
    # Accelerate helps with automatic device mapping for large models
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        token=hf_token,
        device_map="auto"  # Accelerate maneja automáticamente la distribución del modelo
    )
    logger.info("Model loaded successfully")

    # Create pipeline with improved parameters
    logger.info("Creating generation pipeline...")
    model_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.2,
        device_map="auto"
    )
    logger.info("Pipeline created successfully")

except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    raise



@spaces.GPU(duration=60)
@torch.inference_mode()
def clean_response(text):
    """Limpia la respuesta del modelo eliminando etiquetas y texto no deseado"""
    # Eliminar etiquetas INST y wikipedia references
    text = text.replace('[INST]', '').replace('[/INST]', '')
    text = text.replace('(You can find more about it at wikipedia)', '')
    
    # Eliminar cualquier texto que comience con "User:" o "Assistant:"
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if not line.strip().startswith(('User:', 'Assistant:', 'Human:', 'AI:')):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()

@spaces.GPU(duration=60)
@torch.inference_mode()
def generate_response(user_input, chat_history):
    try:
        logger.info("Generating response for user input...")
        global total_water_consumption

        # Calculate water consumption for input
        input_water_consumption = calculate_water_consumption(user_input, True)
        total_water_consumption += input_water_consumption

        # Format conversation history without using INST tags
        formatted_history = ""
        if chat_history:
            for prev_input, prev_response in chat_history:
                formatted_history += f"Question: {prev_input}\nAnswer: {prev_response}\n\n"

        # Create prompt using a más natural format
        prompt = f"""
{system_message}

Previous conversation:
{formatted_history}

Question: {user_input}
Answer:"""

        logger.info("Generating model response...")
        outputs = model_gen(
            prompt,
            max_new_tokens=512,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        # Limpiar y procesar la respuesta
        assistant_response = outputs[0]['generated_text']
        assistant_response = clean_response(assistant_response)
        
        # Si la respuesta sigue conteniendo texto no deseado, intentar extraer solo la parte relevante
        if 'Question:' in assistant_response or 'Answer:' in assistant_response:
            parts = assistant_response.split('Answer:')
            if len(parts) > 1:
                assistant_response = parts[1].split('Question:')[0].strip()
        
        logger.info("Response cleaned and processed")

        # Calculate water consumption for output
        output_water_consumption = calculate_water_consumption(assistant_response, False)
        total_water_consumption += output_water_consumption

        # Update chat history
        chat_history.append([user_input, assistant_response])

        # Update water consumption display
        water_message = f"""
        <div style="position: fixed; top: 20px; right: 20px;
                    background-color: white; padding: 15px;
                    border: 2px solid #2196F3; border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="color: #2196F3; font-size: 24px; font-weight: bold;">
                💧 {total_water_consumption:.4f} ml
            </div>
            <div style="color: #666; font-size: 14px;">
                Water Consumed
            </div>
        </div>
        """

        return chat_history, water_message

    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        error_message = f"I apologize, but I encountered an error. Please try rephrasing your question."
        chat_history.append([user_input, error_message])
        return chat_history, show_water

# Actualizar el system message para ser más específico sobre el formato
system_message = """You are AQuaBot, an AI assistant focused on providing accurate and environmentally conscious information. 

Guidelines for your responses:
1. Provide direct, clear answers without any special tags or markers
2. Do not reference external sources like Wikipedia in your responses
3. Stay focused on the question asked
4. Be concise but informative
5. Be mindful of environmental impact
6. Use a natural, conversational tone

Remember: Never include [INST] tags or other technical markers in your responses."""


# Constants for water consumption calculation
WATER_PER_TOKEN = {
    "input_training": 0.0000309,
    "output_training": 0.0000309,
    "input_inference": 0.05,
    "output_inference": 0.05
}

# Initialize variables
total_water_consumption = 0

def calculate_tokens(text):
    try:
        return len(tokenizer.encode(text))
    except Exception as e:
        logger.error(f"Error calculating tokens: {str(e)}")
        return len(text.split()) + len(text) // 4  # Fallback to approximation

def calculate_water_consumption(text, is_input=True):
    tokens = calculate_tokens(text)
    if is_input:
        return tokens * (WATER_PER_TOKEN["input_training"] + WATER_PER_TOKEN["input_inference"])
    return tokens * (WATER_PER_TOKEN["output_training"] + WATER_PER_TOKEN["output_inference"])

def format_message(role, content):
    return {"role": role, "content": content}

@spaces.GPU(duration=60)
@torch.inference_mode()
def generate_response(user_input, chat_history):
    try:
        logger.info("Generating response for user input...")
        global total_water_consumption

        # Calculate water consumption for input
        input_water_consumption = calculate_water_consumption(user_input, True)
        total_water_consumption += input_water_consumption

        # Create prompt with Llama 2 chat format
        conversation_history = ""
        if chat_history:
            for message in chat_history:
                conversation_history += f"[INST] {message[0]} [/INST] {message[1]} "
        
        prompt = f"<s>[INST] {system_message}\n\n{conversation_history}[INST] {user_input} [/INST]"

        logger.info("Generating model response...")
        outputs = model_gen(
            prompt,
            max_new_tokens=256,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        logger.info("Model response generated successfully")

        assistant_response = outputs[0]['generated_text'].strip()

        # Calculate water consumption for output
        output_water_consumption = calculate_water_consumption(assistant_response, False)
        total_water_consumption += output_water_consumption

        # Update chat history with the new formatted messages
        chat_history.append([user_input, assistant_response])

        # Prepare water consumption message
        water_message = f"""
        <div style="position: fixed; top: 20px; right: 20px;
                    background-color: white; padding: 15px;
                    border: 2px solid #ff0000; border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="color: #ff0000; font-size: 24px; font-weight: bold;">
                💧 {total_water_consumption:.4f} ml
            </div>
            <div style="color: #666; font-size: 14px;">
                Water Consumed
            </div>
        </div>
        """

        return chat_history, water_message

    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        error_message = f"An error occurred: {str(e)}"
        chat_history.append([user_input, error_message])
        return chat_history, show_water

# Create Gradio interface
try:
    logger.info("Creating Gradio interface...")
    with gr.Blocks(css="div.gradio-container {background-color: #f0f2f6}") as demo:
        gr.HTML("""
            <div style="text-align: center; max-width: 800px; margin: 0 auto; padding: 20px;">
                <h1 style="color: #2d333a;">AQuaBot</h1>
                <p style="color: #4a5568;">
                    Welcome to AQuaBot - An AI assistant that helps raise awareness 
                    about water consumption in language models.
                </p>
            </div>
        """)

        chatbot = gr.Chatbot()
        message = gr.Textbox(
            placeholder="Type your message here...",
            show_label=False
        )
        show_water = gr.HTML(f"""
            <div style="position: fixed; top: 20px; right: 20px;
                        background-color: white; padding: 15px;
                        border: 2px solid #ff0000; border-radius: 10px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="color: #ff0000; font-size: 24px; font-weight: bold;">
                    💧 0.0000 ml
                </div>
                <div style="color: #666; font-size: 14px;">
                    Water Consumed
                </div>
            </div>
        """)
        clear = gr.Button("Clear Chat")

        # Add footer with citation and disclaimer
        gr.HTML("""
            <div style="text-align: center; max-width: 800px; margin: 20px auto; padding: 20px;
                        background-color: #f8f9fa; border-radius: 10px;">
                <div style="margin-bottom: 15px;">
                    <p style="color: #666; font-size: 14px; font-style: italic;">
                        Water consumption calculations are based on the study:<br>
                        Li, P. et al. (2023). Making AI Less Thirsty: Uncovering and Addressing the Secret Water
                        Footprint of AI Models. ArXiv Preprint,
                        <a href="https://arxiv.org/abs/2304.03271" target="_blank">https://arxiv.org/abs/2304.03271</a>
                    </p>
                </div>
                <div style="border-top: 1px solid #ddd; padding-top: 15px;">
                    <p style="color: #666; font-size: 14px;">
                        <strong>Important note:</strong> This application uses Meta Llama-2-7b model
                        instead of GPT-3 for availability and cost reasons. However,
                        the water consumption calculations per token (input/output) are based on the
                        conclusions from the cited paper.
                    </p>
                </div>
                <div style="border-top: 1px solid #ddd; margin-top: 15px; padding-top: 15px;">
                    <p style="color: #666; font-size: 14px;">
                        Created by Camilo Vega, AI Consultant 
                        <a href="https://www.linkedin.com/in/camilo-vega-169084b1/" target="_blank">LinkedIn Profile</a>
                    </p>
                </div>
            </div>
        """)

        def submit(user_input, chat_history):
            return generate_response(user_input, chat_history)

        # Configure event handlers
        message.submit(submit, [message, chatbot], [chatbot, show_water])
        clear.click(
            lambda: ([], f"""
                <div style="position: fixed; top: 20px; right: 20px;
                            background-color: white; padding: 15px;
                            border: 2px solid #ff0000; border-radius: 10px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="color: #ff0000; font-size: 24px; font-weight: bold;">
                        💧 0.0000 ml
                    </div>
                    <div style="color: #666; font-size: 14px;">
                        Water Consumed
                    </div>
                </div>
            """),
            None,
            [chatbot, show_water]
        )

    logger.info("Gradio interface created successfully")

    # Launch the application
    logger.info("Launching application...")
    demo.launch()

except Exception as e:
    logger.error(f"Error in Gradio interface creation: {str(e)}")
    raise
