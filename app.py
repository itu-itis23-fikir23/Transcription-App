import streamlit as st
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import torch
import pandas as pd

def load_whisper_model():
    """
    Load the Whisper model for audio transcription.
    """
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")  # Load the Whisper processor
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")  # Load the Whisper model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Select device: GPU if available, else CPU
    model = model.to(device)  # Move the model to the selected device
    return processor, model  

def load_ner_model():
    """
    Load the Named Entity Recognition (NER) model pipeline.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Select device: GPU if available, else CPU
    return pipeline("ner",  # Load the NER pipeline for extracting named entities
                   model="dslim/bert-base-NER",  # Use the pre-trained BERT model for NER
                   device=device,  # Load the model to the selected device (GPU/CPU)
                   aggregation_strategy="simple")  # Aggregate entities in a simple way (no nested entities)

def transcribe_audio(audio_file, processor, model):
    """
    Transcribe audio into text using the Whisper model.
    """
    import soundfile as sf
    import numpy as np
    import io

    audio_bytes = audio_file.read()  # Read the uploaded audio file as bytes
    audio_data, samplerate = sf.read(io.BytesIO(audio_bytes))  # Convert bytes to audio data and get the sample rate
    
    if len(audio_data.shape) > 1:  # Check if the audio is stereo
        audio_data = audio_data.mean(axis=1)  # Convert stereo to mono by averaging the channels

    audio_data = audio_data.astype(np.float32)  # Ensure the audio data is in float32 format
    if audio_data.max() > 1.0:  # Normalize audio if the max value exceeds 1
        audio_data = audio_data / 32768.0  # Normalize to the range [-1, 1]

    if samplerate != 16000:  # If the sample rate is not 16000 Hz, resample
        target_length = int(len(audio_data) * 16000 / samplerate)  # Calculate the target length after resampling
        audio_data = np.interp(  # Interpolate the audio data to the new length
            np.linspace(0, len(audio_data), target_length),
            np.arange(len(audio_data)),
            audio_data
        )
        samplerate = 16000  # Set the samplerate to 16000 Hz

    chunk_duration = 25  # Define the duration of each chunk in seconds
    chunk_size = int(chunk_duration * samplerate)  # Calculate the chunk size in samples
    audio_length = len(audio_data)  # Get the length of the audio data
    transcriptions = []  # Initialize a list to store transcriptions

    # Process audio in chunks of 25 seconds
    for i in range(0, audio_length, chunk_size):
        chunk = audio_data[i:min(i + chunk_size, audio_length)]  # Get the current chunk
        
        input_features = processor(  # Convert the audio chunk to model input features
            chunk, 
            sampling_rate=samplerate,
            return_tensors="pt",
            padding=True  # Padding the input to ensure uniform length
        ).input_features.to(model.device)  # Move the input features to the model's device (GPU/CPU)

        with torch.no_grad():  # Disable gradient calculations 
            predicted_ids = model.generate(  # Generate transcription IDs using the model
                input_features,
                max_length=448,  # Set the maximum length for the output
                num_beams=5,  # Use beam search for better predictions
                forced_decoder_ids=processor.get_decoder_prompt_ids(language="en", task="transcribe")  # Force English transcription
            )

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]  # Decode the IDs into text
        transcriptions.append(transcription)  # Append the transcription to the list

    full_text = " ".join(transcriptions)  # Combine all the transcriptions into a single string
    
    sentences = pd.Series(full_text.split(". "))  # Split the transcription into sentences
    unique_sentences = sentences.drop_duplicates().tolist()  # Remove duplicate sentences
    
    return ". ".join(unique_sentences)  # Join the unique sentences and return the result

def extract_entities(transcription, ner_pipeline):
    """
    Extract entities from transcribed text using the NER model.
    """
    entities = ner_pipeline(transcription)  # Apply the NER pipeline to extract entities
    grouped_entities = {"ORG": set(), "LOC": set(), "PER": set()}  # Initialize a dictionary to store entities by type
    
    # Group entities by type (Organization, Location, Person)
    for entity in entities:
        entity_type = entity["entity_group"]  # Get the entity type (ORG, LOC, PER)
        entity_word = entity["word"].strip()  # Get the entity word and strip any extra spaces
        if entity_type in grouped_entities and len(entity_word) > 1:  # Check if the entity is valid
            # Check if this entity is not a substring of existing entities
            existing = grouped_entities[entity_type]
            if not any(entity_word in ex or ex in entity_word for ex in existing):  # Prevent adding substrings
                grouped_entities[entity_type].add(entity_word)  # Add the entity to the corresponding set

    return {k: sorted(list(v)) for k, v in grouped_entities.items()}  # Return sorted entities by type

def main():

    st.set_page_config(page_title="Meeting Transcription and Entity Extraction", layout="wide")  # Set page layout and title
    st.title("Meeting Transcription and Entity Extraction")  # Set the main title of the app
    
    
    st.markdown("**Name:** Doga Fikir")
    st.markdown("**Student ID:** 150230715")

    # Load the Whisper model and processor for transcription, and NER model for entity extraction
    processor, model = load_whisper_model()
    ner_pipeline = load_ner_model()

    # File uploader for audio input
    uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

    # If the user uploads a file
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")  # Play the uploaded audio file

        # Display a loading spinner while transcribing the audio
        with st.spinner("Transcribing audio..."):
            transcription = transcribe_audio(uploaded_file, processor, model)  # Transcribe the uploaded audio
            st.subheader("Transcription:")  # Display the transcription subtitle
            st.write(transcription)  # Show the transcription result

        # If transcription is successful, extract entities from the transcription
        if transcription:
            with st.spinner("Extracting entities..."):  # Show spinner while extracting entities
                entities = extract_entities(transcription, ner_pipeline)  # Extract entities from transcription

                st.subheader("Extracted Entities:")  # Display subtitle for extracted entities
                col1, col2, col3 = st.columns(3)  # Create 3 columns for different entity types

                # Display the extracted persons (PER) in the first column
                with col1:
                    st.write("### Persons (PERs)")
                    for person in entities["PER"]:
                        st.markdown(f"- {person}")  # List each person entity

                # Display the extracted organizations (ORG) in the second column
                with col2:
                    st.write("### Organizations (ORGs)")
                    for org in entities["ORG"]:
                        st.markdown(f"- {org}")  # List each organization entity

                # Display the extracted locations (LOC) in the third column
                with col3:
                    st.write("### Locations (LOCs)")
                    for loc in entities["LOC"]:
                        st.markdown(f"- {loc}")  # List each location entity

if __name__ == "__main__":
    main()