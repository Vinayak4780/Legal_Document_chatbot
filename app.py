import streamlit as st
import os
from src.retrival import response_with_sources, response_with_sources_streaming
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(
        page_title="Legal Document AI Chatbot",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚖️ Legal Document AI Chatbot")
    st.markdown("Ask questions about Terms & Conditions, Privacy Policies, and Legal Contracts")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Information")
        st.info("Model: Llama3-8B via Groq API")
        st.info("Vector DB: FAISS with BGE embeddings")
        st.info("Retrieval: Top 3 relevant chunks")
        
        # Reset chat button
        if st.button("🔄 Clear Chat", type="primary"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about legal documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
          # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    # Get streaming response and sources
                    result = response_with_sources_streaming(prompt)
                    response_stream = result["response_stream"]
                    sources = result.get("sources", [])
                    
                    # Real token-by-token streaming
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Stream tokens as they arrive from Groq
                    for token in response_stream:
                        full_response += token
                        message_placeholder.markdown(full_response + "▌")
                    
                    # Remove cursor and show final response
                    message_placeholder.markdown(full_response)
                    
                    # Show sources if available
                    if sources:
                        with st.expander("📄 Source Documents"):
                            for i, source in enumerate(sources):
                                st.write(f"**Source {i+1}:** {source}")
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    full_response = "Sorry, I encountered an error processing your request."
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
