import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
from openai import OpenAI

# -------------------- Setup --------------------
st.set_page_config(page_title="🧠 ViT + GPT2 + LLaMA Captioning", layout="centered")
st.title("🖼️ Intelligent Image Captioning App")
st.markdown("Upload an image and generate styled captions using **ViT-GPT2** + **LLaMA-3** 🦙")

# -------------------- Load ViT-GPT2 Model --------------------
@st.cache_resource
def load_vit_gpt2():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, processor, tokenizer

vit_model, vit_processor, tokenizer = load_vit_gpt2()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model.to(device)

# -------------------- Caption Generator --------------------
def generate_captions(img: Image.Image, num_variants: int) -> list:
    if img.mode != "RGB":
        img = img.convert(mode="RGB")

    pixel_values = vit_processor(images=[img] * num_variants, return_tensors="pt").pixel_values.to(device)
    output_ids = vit_model.generate(pixel_values, max_length=16, num_beams=4, num_return_sequences=num_variants)

    captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [c.strip() for c in captions]

# -------------------- LLaMA-3 Rephraser --------------------
@st.cache_resource
def get_llama_client():
    return OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def llama_rephrase_caption(caption, tone="Default", num_variants=1):
    client = get_llama_client()
    
    tone_prompt = {
        "Default": "in a natural tone",
        "Formal": "in a formal tone",
        "Casual": "like you're talking to a friend",
        "Funny": "in a humorous or witty way",
        "Creative": "with a poetic or imaginative flair"
    }

    prompt = f"""
Caption: "{caption}"

Rephrase this caption {num_variants} time(s) {tone_prompt.get(tone, "in a natural tone")}.
Each version should be under 20 words and not too similar.
List them as:
1. ...
2. ...
3. ...
"""

    response = client.chat.completions.create(
        model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    reply = response.choices[0].message.content
    variants = [line.strip()[3:].strip() for line in reply.split("\n") if line.strip().startswith(tuple("123"))]
    return variants if variants else [reply.strip()]

# -------------------- Sidebar Options --------------------
with st.sidebar:
    st.header("⚙️ Caption Settings")
    num_variants = st.selectbox("Number of rephrased variants", [1, 2, 3], index=0)
    tone = st.selectbox("Tone for rephrasing", ["Default", "Formal", "Casual", "Funny", "Creative"])
    generate_button = st.button("🔄 Generate Rephrased Captions")



# -------------------- Main App --------------------
uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Generate base caption immediately
    with st.spinner("🧠 Generating base caption..."):
        base_caption = generate_captions(img, num_variants=1)[0]

    st.markdown("#### 📝 Base Caption:")
    st.markdown(f"> {base_caption}")

    # Only rephrase when the button is clicked
    if generate_button:
        with st.spinner(f"🦙 Rephrasing into {num_variants} variant(s) with tone '{tone}'..."):
            variants = llama_rephrase_caption(base_caption, tone, num_variants=num_variants)

        st.success("✨ Rephrased Caption(s)")
        st.markdown("**🎭 Rephrased Variants:**")
        for i, var in enumerate(variants, 1):
            st.markdown(f"- {var}")

