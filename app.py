import os
import gradio as gr
from moviepy.editor import VideoFileClip
import whisper
import torch
import subprocess
import shutil
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import whisper

model = whisper.load_model("base")

# Load M2M-100 model & tokenizer
m2m_model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(m2m_model_name)
translator_model = M2M100ForConditionalGeneration.from_pretrained(m2m_model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
translator_model.to(device)

TRANSLATION_LANGUAGES = {
    "English (No Translation)": "en",
    "Urdu": "ur",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Chinese": "zh",
    "Arabic": "ar",
    "Hindi": "hi"
}

def translate_text_m2m(text_list, target_lang):
    """Translates a list of English texts into the target language using M2M-100."""
    if target_lang == "en":
        return text_list  # No translation needed
    
    tokenizer.src_lang = "en"
    inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = translator_model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
    
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def generate_translated_subtitles(video_path, target_language):
    """Extracts audio, transcribes it with Whisper, translates subtitles, and saves an SRT file."""
    video = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path)
    
    # Transcribe with Whisper
    result = model.transcribe(audio_path, language="en")
    os.remove(audio_path)
    
    texts = [segment['text'] for segment in result['segments']]
    translated_texts = translate_text_m2m(texts, TRANSLATION_LANGUAGES[target_language])
    
    srt_filename = f"subtitles_{TRANSLATION_LANGUAGES.get(target_language, 'en')}.srt"
    
    # UTF-8 encoding
    with open(srt_filename, "w", encoding="utf-8-sig") as srt_file:
        for index, (segment, translated_text) in enumerate(zip(result['segments'], translated_texts)):
            start_time, end_time = segment['start'], segment['end']

            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                seconds = seconds % 60
                milliseconds = int((seconds - int(seconds)) * 1000)
                return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"

            srt_file.write(f"{index + 1}\n")
            srt_file.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
            srt_file.write(f"{translated_text}\n\n")

    return srt_filename

def burn_subtitles_on_video(video_path, srt_path):
    """Uses ffmpeg to burn subtitles into the video."""
    new_video_path = "input_video.mp4"
    new_srt_path = "subtitles.srt"
    output_video = "video_with_subtitles.mp4"

    shutil.copy(video_path, new_video_path)
    shutil.copy(srt_path, new_srt_path)

    command = [
    "ffmpeg",
    "-y",
    "-i", "input_video.mp4",
    "-vf", "subtitles=subtitles.srt:force_style='Fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf,Fontsize=24,PrimaryColour=&HFFFFFF&'",
    "-c:a", "copy",
    "video_with_subtitles.mp4"
    ]

   
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("FFmpeg Output:", result.stdout)
        print("FFmpeg Error:", result.stderr)
        return output_video
    except subprocess.CalledProcessError as e:
        print("FFmpeg Error:", e.stderr)
        return None 


def video_to_translated_subtitles(video, target_language, output_type):
    """Processes video: generates subtitles, translates (if needed), burns subtitles, and returns files."""
    srt_filename = generate_translated_subtitles(video, target_language)
    if output_type == "SRT File":
        return srt_filename, srt_filename
    burned_video = burn_subtitles_on_video(video, srt_filename)
    return burned_video, burned_video

iface = gr.Interface(
    fn=video_to_translated_subtitles,
    inputs=[
        gr.Video(label="Upload English Video"),
        gr.Dropdown(choices=list(TRANSLATION_LANGUAGES.keys()), label="Translate to", value="English (No Translation)"),
        gr.Radio(["SRT File", "Burned-in Subtitles"], label="Select Output Type"),
    ],
    outputs=[
        gr.File(label="Output"),
        gr.DownloadButton(label="Download File")
    ],
    title="Video to Subtitles (With Translation)",
    description="Upload an English video, and get subtitles in your desired language"
)

iface.launch(share=True)