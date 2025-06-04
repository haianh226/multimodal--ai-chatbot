import gradio as gr
import os
from dotenv import load_dotenv
from modules.groq_inference import GroqMultimodalProcessor

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize processor
if GROQ_API_KEY:
    processor = GroqMultimodalProcessor(GROQ_API_KEY)
else:
    processor = None

def gradio_multimodal_interface(text_input, image_input, audio_input):
    """Main function for Gradio interface"""
    if not processor:
        return "❌ Error: GROQ_API_KEY not found. Please check your .env file."
    
    try:
        response = processor.multimodal_chat(text_input, image_input, audio_input)
        return response
    except Exception as e:
        return f"❌ Error: {str(e)}"

def create_gradio_app():
    """Create and configure Gradio app"""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .gr-button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        border: none;
        color: white;
        font-weight: bold;
    }
    .gr-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    """
    
    with gr.Blocks(css=custom_css, title="🤖 Multimodal AI Chatbot") as demo:
        
        # Header
        gr.Markdown("""
        # 🤖 Multimodal AI Chatbot with Groq
        
        **Welcome to Multimodal AI Chatbot!** 
        
        Here you can:
        - 💬 Enter text
        - 🖼️ Upload image
        - 🎵 Record or download audio file
        - 🚀 Do it all at once!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Your input")
                
                # Text input
                text_box = gr.Textbox(
                    label="💬 Enter a question or content",
                    placeholder="For example: Explain the picture...",
                    lines=3
                )
                
                # Image input
                image_uploader = gr.Image(
                    type="filepath",
                    label="🖼️ Upload image (optional)",
                    height=200
                )
                
                # Audio input
                audio_recorder = gr.Audio(
                    type="filepath",
                    label="🎵 Record or upload auido file (optional)"
                )
                
                # Submit button
                chat_button = gr.Button(
                    "🚀 Send request", 
                    variant="primary",
                    size="lg"
                )
                
                # Clear button
                clear_button = gr.Button(
                    "🗑️ Delete all",
                    variant="secondary"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 🤖 AI respond")
                
                # Output
                output_box = gr.Textbox(
                    label="💡 Answer",
                    lines=15,
                    max_lines=20,
                    show_copy_button=True
                )
                
                # Status
                status_box = gr.Textbox(
                    label="📊 Status",
                    lines=2,
                    interactive=False
                )
        
        # Examples section
        gr.Markdown("### 💡How to use ")
        
        gr.Examples(
            examples=[
                ["Hello! What can you do?", None, None],
                ["Write a short poem ", None, None],
                ["Explain machine learning", None, None],
            ],
            inputs=[text_box, image_uploader, audio_recorder],
        )
        
        # Event handlers
        def process_and_update_status(text, image, audio):
            if not processor:
                return "❌ Error: API key not found", "❌ Missing configuration"
            
            # Update status
            status = "🔄 Processing"
            if text: status += " • Text ✓"
            if image: status += " • Picture ✓" 
            if audio: status += " • Audio ✓"
            
            try:
                response = processor.multimodal_chat(text, image, audio)
                final_status = "✅ Completed"
                return response, final_status
            except Exception as e:
                error_status = f"❌ Error: {str(e)}"
                return f"Sorry, an error occurred: {str(e)}", error_status
        
        chat_button.click(
            fn=process_and_update_status,
            inputs=[text_box, image_uploader, audio_recorder],
            outputs=[output_box, status_box]
        )
        
        # Clear function
        def clear_all():
            return "", None, None, "", "🔄 Deleted all"
        
        clear_button.click(
            fn=clear_all,
            outputs=[text_box, image_uploader, audio_recorder, output_box, status_box]
        )
        
        # Footer
        gr.Markdown("""
        ---
        **Created by:** Hai Anh - Multimodal Chatbot  
        **Technology:** Groq API + Gradio + Python  
        """)
    
    return demo

def main():
    """Main function to launch the app"""
    print("🚀 Starting Multimodal Chatbot...")
    
    if not GROQ_API_KEY:
        print("❌ Error: GROQ_API_KEY not found in .env file")
        print("Please create a .env file with your Groq API key")
        return
    
    print("✅ API Key loaded successfully")
    print("🌐 Creating Gradio interface...")
    
    demo = create_gradio_app()
    
    print("🎉 Launching app...")
    print("📱 Access the app at: http://127.0.0.1:7860")
    
    # Launch with custom settings
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # Set to True if you want public link
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main()