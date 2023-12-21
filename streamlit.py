from main import ChatBot
import streamlit as st

bot = ChatBot()
    
st.set_page_config(page_title="Random Horoscope Bot")
with st.sidebar:
    st.title('Random Horoscope Bot')

# Function for generating LLM response
def generate_response(input):
    result = bot.qa_chain(input)
    res = result["result"]
    return res

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    print('hello')
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(input) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)