"""
Streamlit Web Application for Image Captioning Model Comparison.

This script creates a web interface using Streamlit to compare the performance
of different image captioning models. Users can upload an image, and the
application will generate captions using four distinct models:
1.  A custom-built LSTM model.
2.  A custom-built Bidirectional LSTM (BiLSTM) model.
3.  A pre-trained Vision Transformer (ViT) combined with GPT-2 (from Hugging Face).
4.  A pre-trained BLIP (Bootstrapping Language-Image Pre-training) model (from Hugging Face).

The application handles:
- Loading necessary models and tokenizers (using caching for efficiency).
- Preprocessing the uploaded image (resizing and feature extraction).
- Generating captions using each loaded model.
- Displaying the uploaded image and the generated captions side-by-side.
- Optionally translating the generated English captions into Arabic using googletrans.
- Error handling for model loading and caption generation.
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from googletrans import Translator # Library for translation
import pickle # For loading the custom tokenizer
import os # For checking file paths
import torch # PyTorch is required for Hugging Face Transformers
import requests # Potentially used by some libraries, good to keep

# --- Hugging Face Transformers Imports ---
# Used for loading pre-trained models like ViT-GPT2 and BLIP
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- TensorFlow Keras Imports for Custom Models ---
from tensorflow.keras.applications import DenseNet201 # Feature extractor for custom models
from tensorflow.keras.preprocessing.sequence import pad_sequences # For padding text sequences
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Bidirectional, concatenate # Layers for custom models
from tensorflow.keras.models import Model # Keras Model class

# --- Streamlit Page Configuration ---
# Set the page layout to wide for better use of screen space
st.set_page_config(layout="wide")

# --- Components for Custom Models (LSTM, BiLSTM) ---

@st.cache_resource # Cache the loaded feature extractor model to avoid reloading on every interaction
def load_feature_extractor():
    """Loads the DenseNet201 model pre-trained on ImageNet, configured for feature extraction."""
    # include_top=False removes the classification layer
    # pooling='avg' adds a global average pooling layer
    # input_shape specifies the expected image dimensions
    return DenseNet201(include_top=False, weights='imagenet', pooling='avg', input_shape=(224, 224, 3))

# Load the feature extractor model
feat_model_custom = load_feature_extractor()

@st.cache_resource # Cache the loaded tokenizer
def load_custom_tokenizer(path):
    """Loads the custom tokenizer object saved using pickle."""
    if not os.path.exists(path):
        st.error(f"Error: Tokenizer file not found at path: {path}.")
        return None
    try:
        with open(path, 'rb') as f:
            tokenizer_custom = pickle.load(f)
        return tokenizer_custom
    except Exception as e:
        st.error(f"An error occurred while loading the tokenizer: {e}")
        return None

# Define the path to the saved tokenizer file (ensure this path is correct)
tokenizer_path = '/content/drive/MyDrive/tokenizer.pkl' # Make sure this path is correct for your environment
tokenizer = load_custom_tokenizer(tokenizer_path)

# Stop the app if the tokenizer failed to load
if tokenizer is None:
    st.error("Tokenizer could not be loaded. Stopping the application.")
    st.stop()

# --- Custom Model Parameters ---
# Vocabulary size: number of unique words + 1 for the padding token (usually 0)
vocab_size = len(tokenizer.word_index) + 1
# Maximum length of captions used during training (important for padding)
max_length = 37

# --- Custom Model Building Functions ---

def build_model1(feature_size, vocab_size, max_length):
    """Builds the custom LSTM image captioning model architecture."""
    # Input layer for image features (output from DenseNet201)
    image_input = Input(shape=(feature_size,))
    # Dense layer to process image features, reducing dimensionality
    img_feats = Dense(256, activation='relu')(image_input)

    # Input layer for the caption sequence (padded)
    caption_input = Input(shape=(max_length,))
    # Embedding layer to convert word indices into dense vectors
    # mask_zero=True tells subsequent layers to ignore padding
    emb = Embedding(vocab_size, 256, mask_zero=True)(caption_input)
    # LSTM layer to process the sequence of word embeddings
    lstm = LSTM(256)(emb) # Returns only the final hidden state

    # Concatenate image features and LSTM output
    combined = concatenate([img_feats, lstm])
    # Final Dense layer with softmax activation to predict the next word index
    output = Dense(vocab_size, activation='softmax')(combined)

    # Define the full model
    model = Model(inputs=[image_input, caption_input], outputs=output)
    # Compile the model (optimizer and loss used during training, not essential for inference only but good practice)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def build_model2(feature_size, vocab_size, max_length):
    """Builds the custom Bidirectional LSTM (BiLSTM) image captioning model architecture."""
    # Input layer for image features
    image_input = Input(shape=(feature_size,))
    # Dense layer for image features
    img_feats = Dense(256, activation='relu')(image_input)

    # Input layer for the caption sequence
    caption_input = Input(shape=(max_length,))
    # Embedding layer
    emb = Embedding(vocab_size, 256, mask_zero=True)(caption_input)
    # Bidirectional LSTM layer processes the sequence in both forward and backward directions
    bilstm = Bidirectional(LSTM(128))(emb) # Output dimension is 2 * 128 = 256

    # Concatenate image features and BiLSTM output
    combined = concatenate([img_feats, bilstm])
    # Final prediction layer
    output = Dense(vocab_size, activation='softmax')(combined)

    # Define the full model
    model = Model(inputs=[image_input, caption_input], outputs=output)
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# --- Load Weights for Custom Models (LSTM, BiLSTM) ---

# Define the feature size expected from DenseNet201's average pooling layer
feature_size_custom = 1920

# Define paths to the saved model weight files (or full .keras models if saved that way)
# Ensure these paths are correct for your environment
model1_path = '/content/drive/MyDrive/ImageCaptioningModels_Org/model1_lstm_full_model.keras'
model2_path = '/content/drive/MyDrive/ImageCaptioningModels_Org/model2_bilstm_full_model.keras'
# Transformer model path removed as per user request

# Flag to track if custom models are loaded successfully
custom_models_loaded = True
# Dictionary to store the loaded custom models
loaded_custom_models = {}

try:
    @st.cache_resource # Cache the loaded models with weights
    def load_custom_model_weights(_model_builder, weights_path, *args):
        """Builds a model using the builder and loads weights from the given path."""
        # Build the model architecture first
        model = _model_builder(*args)
        if os.path.exists(weights_path):
            # Load the saved weights into the model structure
            # Use model.load_weights if only weights were saved,
            # or load_model if the entire model was saved (adjust accordingly)
            # Assuming .keras files contain the full model:
            # model = tf.keras.models.load_model(weights_path) # Use this if you saved the full model
            model = tf.keras.models.load_model(weights_path) # Use this if you only saved weights
            return model
        else:
            st.warning(f"Weight file not found: {weights_path}")
            return None

    # Load Model 1 (LSTM)
    model1 = load_custom_model_weights(build_model1, model1_path, feature_size_custom, vocab_size, max_length)
    if model1:
        loaded_custom_models['Model 1 (LSTM - Custom)'] = model1
    else:
        # If loading fails, set the flag to False
        custom_models_loaded = False
        st.error("Failed to load Model 1 (LSTM - Custom).")


    # Load Model 2 (BiLSTM)
    model2 = load_custom_model_weights(build_model2, model2_path, feature_size_custom, vocab_size, max_length)
    if model2:
        loaded_custom_models['Model 2 (BiLSTM - Custom)'] = model2
    else:
        # If loading fails, set the flag to False
        custom_models_loaded = False
        st.error("Failed to load Model 2 (BiLSTM - Custom).")

    # Display success or error messages based on loading status
    if custom_models_loaded and len(loaded_custom_models) == 2: # Check if both models loaded
        st.success("Custom models (LSTM, BiLSTM) loaded successfully.")
    elif not loaded_custom_models:
         st.error("Failed to load any custom models.")
         custom_models_loaded = False # Ensure flag reflects complete failure

except Exception as e:
    st.error(f"An error occurred while building or loading custom model weights: {e}")
    custom_models_loaded = False

# 1. Load ViT-GPT2 (Model 3)
vit_gpt2_model_loaded = False
try:
    @st.cache_resource # Cache the loaded Hugging Face model and components
    def load_vit_gpt2_model():
        """Loads the ViT-GPT2 model, processor, and tokenizer from Hugging Face."""
        model_name = "nlpconnect/vit-gpt2-image-captioning"
        # Load the main encoder-decoder model
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        # Load the image processor (handles image preprocessing for ViT)
        processor = ViTImageProcessor.from_pretrained(model_name)
        # Load the tokenizer (handles text processing for GPT-2 decoder)
        tokenizer_hf = AutoTokenizer.from_pretrained(model_name) # Use a different variable name
        # Determine device (use GPU if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Move the model to the selected device
        model.to(device)
        st.success(f"ViT-GPT2 model ({model_name}) loaded successfully on {device}.")
        return model, processor, tokenizer_hf, device

    # Execute the loading function
    vit_model, vit_processor, vit_tokenizer, vit_device = load_vit_gpt2_model()
    vit_gpt2_model_loaded = True

except Exception as e:
    st.error(f"An error occurred while loading the ViT-GPT2 model: {e}")
    st.warning("Caption generation using Model 3 (ViT-GPT2) will be unavailable.")

# 2. Load BLIP (Model 4)
blip_model_loaded = False
try:
    @st.cache_resource # Cache the loaded BLIP model and processor
    def load_blip_model():
        """Loads the BLIP model and processor from Hugging Face."""
        model_name = "Salesforce/blip-image-captioning-base"
        # Load the processor (handles both image and text preprocessing for BLIP)
        processor = BlipProcessor.from_pretrained(model_name)
        # Load the main conditional generation model
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Move model to device
        model.to(device)
        st.success(f"BLIP model ({model_name}) loaded successfully on {device}.")
        return model, processor, device

    # Execute the loading function
    blip_model, blip_processor, blip_device = load_blip_model()
    blip_model_loaded = True

except Exception as e:
    st.error(f"An error occurred while loading the BLIP model: {e}")
    st.warning("Caption generation using Model 4 (BLIP) will be unavailable.")


# --- Helper Functions for Caption Generation ---

# For Custom Models (LSTM, BiLSTM)
@st.cache_data # Cache the results of feature extraction for a given image
def extract_features_custom(img_pil):
    """Extracts image features using the loaded DenseNet201 model."""
    try:
        # Resize image to the expected input size (224x224)
        img = img_pil.resize((224, 224))
        # Convert PIL image to numpy array
        img_array = np.array(img)
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        # Preprocess the image array according to DenseNet requirements
        img_array = tf.keras.applications.densenet.preprocess_input(img_array)

        if feat_model_custom:
            # Get features from the DenseNet model
            features = feat_model_custom.predict_on_batch(img_array) # Use predict_on_batch for single batch
            return features.squeeze() # Remove single dimensions
        else:
            st.error("Custom feature extractor (DenseNet201) not initialized.")
            return None
    except Exception as e:
        st.error(f"Error during custom feature extraction: {e}")
        return None

def word_for_id(integer, tokenizer_custom):
    """Converts a word index back into a word string using the tokenizer."""
    for word, index in tokenizer_custom.word_index.items():
        if index == integer:
            return word
    return None # Return None if index not found

def generate_caption_custom(model, tokenizer_custom, photo_feature, max_len):
    """Generates a caption using a custom LSTM or BiLSTM model via greedy search."""
    if photo_feature is None:
        return "Error during feature extraction."

    # Start the sequence with the start token
    in_text = 'startseq'
    try:
        # Generate words step-by-step up to max_len
        for _ in range(max_len):
            # Convert the current sequence text to indices
            sequence = tokenizer_custom.texts_to_sequences([in_text])[0]
            # Pad the sequence to the maximum length
            sequence = pad_sequences([sequence], maxlen=max_len, padding='post')
            # Predict the next word probabilities
            # Reshape photo_feature to add batch dimension (1, feature_size)
            yhat = model.predict_on_batch([photo_feature.reshape((1, -1)), sequence])
            # Get the index of the word with the highest probability (greedy search)
            yhat = np.argmax(yhat)
            # Convert the index back to a word
            word = word_for_id(yhat, tokenizer_custom)
            # Stop if the word is None (index not found) or the end token
            if word is None or word == 'endseq':
                break
            # Append the predicted word to the sequence
            in_text += ' ' + word

        # Remove the initial 'startseq' token from the final caption
        caption = in_text.split(' ', 1)[1] if ' ' in in_text else ''
        # Clean up any trailing 'endseq' if it was somehow included
        return caption.replace(' endseq', '')
    except Exception as e:
        print(f"Error generating custom caption: {e}") # Log error for debugging
        return "Failed to generate custom caption."

# For ViT-GPT2 Model (Model 3)
def generate_caption_vit_gpt2(model_obj, processor_obj, tokenizer_obj, img_pil, device, max_len=20, num_beams=4):
    """Generates a caption using the pre-trained ViT-GPT2 model."""
    try:
        # Preprocess the image using the ViTImageProcessor
        pixel_values = processor_obj(images=[img_pil], return_tensors="pt").pixel_values
        # Move pixel values tensor to the appropriate device
        pixel_values = pixel_values.to(device)

        # Set generation parameters (beam search is often used for better quality)
        gen_kwargs = {"max_length": max_len, "num_beams": num_beams}
        # Generate caption IDs using the model
        output_ids = model_obj.generate(pixel_values, **gen_kwargs)

        # Decode the generated IDs back into text using the tokenizer
        preds = tokenizer_obj.batch_decode(output_ids, skip_special_tokens=True)
        # Clean up whitespace
        preds = [pred.strip() for pred in preds]
        return preds[0] if preds else "Generation failed"
    except Exception as e:
        print(f"Error generating ViT-GPT2 caption: {e}") # Log error
        return "Failed to generate caption (ViT-GPT2)."

# For BLIP Model (Model 4)
def generate_caption_blip(model_obj, processor_obj, img_pil, device, max_len=30, num_beams=4):
    """Generates a caption using the pre-trained BLIP model."""
    try:
        # Preprocess the image using the BlipProcessor
        inputs = processor_obj(images=img_pil, return_tensors="pt").to(device)
        # Set generation parameters
        gen_kwargs = {"max_length": max_len, "num_beams": num_beams}
        # Generate caption IDs
        output_ids = model_obj.generate(**inputs, **gen_kwargs)
        # Decode the IDs back to text
        caption = processor_obj.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        print(f"Error generating BLIP caption: {e}") # Log error
        return "Failed to generate caption (BLIP)."

# === Translator Setup ===
try:
    # Initialize the translator object
    translator = Translator()
except Exception as e:
    # Handle potential errors during translator initialization (e.g., network issues)
    st.error(f"Failed to initialize translator: {e}. Translation will be unavailable.")
    translator = None

# === Streamlit User Interface ===

st.title(" Image Captioning Models Comparison ")
st.markdown("Upload an image to generate captions using: Custom LSTM, Custom BiLSTM, Pre-trained ViT-GPT2, and Pre-trained BLIP.")

# File uploader widget allows users to select an image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

# Process only if a file has been uploaded
if uploaded_file is not None:
    # Create two columns for layout: one for the image, one for captions
    col1, col2 = st.columns([1, 2]) # Image column is narrower

    with col1:
        # Open the uploaded image using PIL and convert to RGB
        img_pil = Image.open(uploaded_file).convert('RGB')
        # Display the image
        st.image(img_pil, caption="🖼 Uploaded Image", use_column_width=True)

    with col2:
        st.subheader("⏳ Generating Captions...")
        # List to store results from all models
        captions_data = []

        # --- Generate Captions for Custom Models (1 & 2) ---
        if custom_models_loaded and loaded_custom_models:
            # Extract features only once for both custom models
            photo_feat_custom = extract_features_custom(img_pil)
            if photo_feat_custom is not None:
                # Use a spinner to indicate processing
                with st.spinner('Generating captions from custom models (LSTM, BiLSTM)...'):
                    # Loop through the dictionary of loaded custom models
                    for model_name, model_obj in loaded_custom_models.items():
                        # Generate caption using the specific custom model
                        cap = generate_caption_custom(model_obj, tokenizer, photo_feat_custom, max_length)
                        # Store the result
                        captions_data.append({"model": model_name, "caption": cap})
            else:
                # Handle feature extraction errors
                st.error("Failed to extract features for custom models.")
                if 'Model 1 (LSTM - Custom)' in loaded_custom_models: captions_data.append({"model": "Model 1 (LSTM - Custom)", "caption": "Feature Extraction Error"})
                if 'Model 2 (BiLSTM - Custom)' in loaded_custom_models: captions_data.append({"model": "Model 2 (BiLSTM - Custom)", "caption": "Feature Extraction Error"})
        else:
             # Handle cases where custom models failed to load
             st.warning("Some or all custom models were not loaded.")
             if 'Model 1 (LSTM - Custom)' not in loaded_custom_models: captions_data.append({"model": "Model 1 (LSTM - Custom)", "caption": "Not Loaded"})
             if 'Model 2 (BiLSTM - Custom)' not in loaded_custom_models: captions_data.append({"model": "Model 2 (BiLSTM - Custom)", "caption": "Not Loaded"})


        # --- Generate Caption for ViT-GPT2 Model (Model 3) ---
        if vit_gpt2_model_loaded:
            with st.spinner('Generating caption from ViT-GPT2 model...'):
                # Generate caption using the ViT-GPT2 helper function
                vit_caption = generate_caption_vit_gpt2(vit_model, vit_processor, vit_tokenizer, img_pil, vit_device)
                captions_data.append({"model": "Model 3 (ViT-GPT2 Pre-trained)", "caption": vit_caption})
        else:
            # Handle case where ViT-GPT2 failed to load
            st.warning("ViT-GPT2 model was not loaded.")
            captions_data.append({"model": "Model 3 (ViT-GPT2 Pre-trained)", "caption": "Not Loaded / Error"})

        # --- Generate Caption for BLIP Model (Model 4) ---
        if blip_model_loaded:
            with st.spinner('Generating caption from BLIP model...'):
                # Generate caption using the BLIP helper function
                blip_caption = generate_caption_blip(blip_model, blip_processor, img_pil, blip_device)
                captions_data.append({"model": "Model 4 (BLIP Pre-trained)", "caption": blip_caption})
        else:
            # Handle case where BLIP failed to load
            st.warning("BLIP model was not loaded.")
            captions_data.append({"model": "Model 4 (BLIP Pre-trained)", "caption": "Not Loaded / Error"})

        # --- Display Results ---
        st.subheader("📜 Generated Captions")
        if not captions_data:
             st.warning("No captions were generated.") # Message if something went wrong entirely

        # Iterate through the collected caption data and display each result
        for data in captions_data:
            st.markdown(f"---") # Separator line
            st.markdown(f"##### {data['model']}") # Model name as sub-header
            st.write(f"{data['caption']}") # Display the generated caption

            # --- Optional Translation ---
            # Check if translator is available, caption is a valid string, and not an error message
            if translator and isinstance(data['caption'], str) and not data['caption'].lower().startswith(("error", "not loaded", "failed", "فشل")):
                try:
                   # Show spinner during translation API call
                   with st.spinner(f"Translating caption from {data['model']}..."):
                       # Translate the caption to Arabic ('ar')
                       translation = translator.translate(data['caption'], dest='ar')
                       st.write(f"Arabic: {translation.text}") # Display translated text
                except Exception as e:
                    # Handle translation errors
                    st.write("Translation to Arabic failed.")
            elif not translator:
                # Inform user if translator is unavailable
                st.write("Translation is currently disabled.")
            elif not isinstance(data['caption'], str) or data['caption'].lower().startswith(("error", "not loaded", "failed")):
                # Indicate why translation is skipped for error/non-string captions
                st.write("*Cannot translate this caption (Error or Not Loaded).")