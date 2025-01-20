import streamlit as st
import torch
from model import load_model, generate_text

# Page config
st.set_page_config(
    page_title="GPT Text Generator",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def get_model():
    model = load_model("gpt-config.pth")
    return model

def main():
    st.title("🤖 GPT Text Generator")
    st.write("Generate text using a custom GPT model")

    # Sidebar controls
    st.sidebar.header("Generation Parameters")
    max_length = st.sidebar.slider("Max Length", 10, 200, 50)
    num_sequences = st.sidebar.slider("Number of Sequences", 1, 5, 3)
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0)
    
    # Main interface
    prompt = st.text_area("Enter your prompt:", height=100)
    
    if st.button("Generate"):
        if prompt:
            try:
                with st.spinner("Generating text..."):
                    model = get_model()
                    generated_texts = generate_text(
                        text=prompt,
                        model=model,
                        max_length=max_length,
                        num_return_sequences=num_sequences,
                        temperature=temperature
                    )
                
                st.subheader("Generated Text:")
                for i, text in enumerate(generated_texts, 1):
                    st.markdown(f"**Version {i}:**")
                    st.write(text)
                    st.divider()
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a prompt first!")

    # Add information about the model
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Information")
    st.sidebar.markdown("""
    - Architecture: GPT-2
    - Layers: 12
    - Heads: 12
    - Embedding Dim: 768
    - Vocab Size: 50,257
    """)

if __name__ == "__main__":
    main() 