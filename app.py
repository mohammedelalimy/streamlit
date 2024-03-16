import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig

local_model_dir = "peft_model/"
HUGGING_FACE_USER_NAME = "elalimy"
model_name = "my_awesome_peft_finetuned_helsinki_model"
peft_model_id = f"{HUGGING_FACE_USER_NAME}/{model_name}"
# Load model configuration (assuming it's saved locally)
config = PeftConfig.from_pretrained(local_model_dir, local_files_only=True)
# Load the base model from its local directory (replace with actual model type)
model = AutoModelForSeq2SeqLM.from_pretrained(
    local_model_dir, return_dict=True, load_in_8bit=False)
# Load the tokenizer from its local directory (replace with actual tokenizer type)
tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True)
# Load the Peft model (assuming it's a custom class or adaptation)
AI_model = PeftModel.from_pretrained(model, peft_model_id)


def generate_translation(model, tokenizer, source_text, device="cpu"):
    # Encode the source text
    input_ids = tokenizer.encode(source_text, return_tensors='pt').to(device)

    # Move the model to the same device as input_ids
    model = model.to(device)

    # Generate the translation with adjusted decoding parameters
    generated_ids = model.generate(
        input_ids=input_ids,
        max_length=512,  # Adjust max_length if needed
        num_beams=4,
        length_penalty=5,  # Adjust length_penalty if needed
        no_repeat_ngram_size=4,
        early_stopping=True
    )

    # Decode the generated translation excluding special tokens
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text


def main():
    st.title("Arabic to English Translation")

    # Get text input from user
    text_to_translate = st.text_area("Enter Arabic text:", "")

    # Translate text when user clicks the "Translate" button
    if st.button("Translate"):
        translated_text = generate_translation(AI_model, tokenizer, text_to_translate)
        st.write("Translated text:")
        st.write(translated_text)


if __name__ == "__main__":
    main()
