import io
from openai import OpenAI
import streamlit as st
from io import BytesIO
from huggingface_hub import InferenceClient
import requests
st.set_page_config(page_title='MULTICHAT AI')
if "mode" not in st.session_state:
    st.session_state.mode = "chat"
if "messages" not in st.session_state:
    st.session_state.messages = []
def export_bytes(messages):
    text = "========== CHAT HISTORY ==========\n\n"

    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        text += f"{role}: {message['content']}\n"

    text += "========== END OF CHAT ==========\n"

    return io.BytesIO(text.encode("utf-8"))


with st.sidebar:
    st.header("Settings")
    if st.session_state.mode=='chat':
        image=st.sidebar.button('Image Generation')
        if image:
                st.session_state.mode='image'
                st.rerun()
    elif st.session_state.mode=='image':
        chat=st.sidebar.button('Chatbot')
        if chat:
                st.session_state.mode='chat'
                st.rerun()
    if st.session_state.mode=='chat':            
        groq_api_key=st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_..."
        ).strip()
    elif st.session_state.mode=='image':
        hf_api_key=st.text_input(
            "Hugging Face API Key",
            type="password",
            placeholder="hf_..."
        ).strip()

    temp = st.slider(
        "Temperature",
        0.0,
        1.0,
        0.7,
        0.1,
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    download_placeholder = st.sidebar.empty()



if st.session_state.mode=='chat':
    st.title("Chatbot")
    if not groq_api_key:
        st.info("<--- Please enter your Groq API key to start chatting.")
        st.stop()
    client = OpenAI(api_key=groq_api_key,
                    base_url="https://api.groq.com/openai/v1")
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "llama-3.3-70b-versatile"
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                temperature=temp,
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
        if st.session_state.messages:
                download_placeholder.download_button(
                    "Export chat history",
                    data=export_bytes(st.session_state.messages),
                    file_name="chat_history.txt",
                    mime="text/plain",
                )
elif st.session_state.mode=='image':
    MODEL_ID='stabilityai/stable-diffusion-3-medium-diffusers'
    FILTER_API_URL='https://filters-zeta.vercel.app/api/filter'
    ENHANCE_SYS=(
    "Improve prompts for text-to-image. Return ONLY the enhanced prompt. "
    "Add subject, style, lighting, camera angle, background, colors. Keep it safe."
    )
    NEGATIVE='low quality, blurry, distorted, watermark, text, cropped'
    
    img_client=InferenceClient(provider='hf-inference',api_key=hf_api_key)
    def check_prompt_with_filter(prompt:str):
        try:
            response=requests.post(
                FILTER_API_URL,json={'prompt':prompt},timeout=10
            )
            response.raise_for_status()
            data=response.json()
            if not isinstance(data,dict):
                return {'ok':False,'reason':'Invalid filter API response!'}
            return data
        except Exception as e:
            return {'ok':False,'reason':f'API ERROR-{str(e)}'}
    def enhance(raw:str)->str:
        from hf import generate_response
        out=generate_response(
            f'{ENHANCE_SYS}\nUser prompt: {raw}',temperature=temp,max_tokens=400
        )
        return (out or raw).strip()
    def gen_image(prompt:str):
        filter_result=check_prompt_with_filter(prompt)
        if not filter_result.get('ok'):
            return None
        try:
            return img_client.text_to_image(prompt=prompt,negative_prompt=NEGATIVE,model=MODEL_ID),None
        except Exception as e:
            msg=str(e)
            if 'negative_prompt' in msg or 'unexpected keyword'in msg:
                try:
                    return img_client.text_to_image(prompt=prompt,negative_prompt=NEGATIVE,model=MODEL_ID),None
                except Exception as e2:
                    msg=str(e2)
            if any(x in msg for x in ["402", "Payment Required", "pre-paid credits"]):
                return None, "Image backend requires credits or model not available on hf-inference.\n\nRaw error: " + msg
            if "404" in msg or "Not Found" in msg:
                return None, "Model not served on this provider route (hf-inference).\n\nRaw error: " + msg
            return None, "Error during image generation: " + msg
    def main():
        st.title('AI image generator')
        if not hf_api_key:
                st.info('Enter you Hugging Face API key in the sidebar!')
                st.stop()
        st.info('FLOW: Enter a prompt -> Enhance it -> Check it using a safety API -> Generate image')
        with st.form('image_form'):
            raw=st.text_area(
                'Image description',
                height=120,
                placeholder='Example:A cosy cabin on a snowy mountain.'
            )
            submit=st.form_submit_button('Generate image!')
        if submit:
            raw=raw.strip()
            if not raw:
                st.warning('PLEASE ENTER A IMAGE DESCRIPTION!')
                return
            raw_check=check_prompt_with_filter(raw)
            if not raw_check.get('ok'):
                st.error('Prompt locked!')
                return
            with st.spinner('Enchaning prompt...'):
                final_prompt=enhance(raw)
            enhance_check=check_prompt_with_filter(final_prompt)
            if not enhance_check.get('ok'):
                st.error('Enhanced promt LOCKED!')
                return
            st.markdown('Enchanced prompt')
            st.code(final_prompt)
            with st.spinner('Generating image'):
                img,err=gen_image(final_prompt)
            if err:
                st.error(err)
                return
            st.image(img,caption='Generated image',use_container_width=True)
            st.session_state.generated_image=img
        img=st.session_state.get('generated_image')
        if img:
            buf = BytesIO()

            img.save(buf, format="PNG")

            st.download_button(

            "📥 Download Image",

            buf.getvalue(),

            "ai_generated_image.png",

            "image/png")
    main()

