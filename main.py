import os
import openai
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from google.cloud import speech
from pydub import AudioSegment
from dotenv import load_dotenv
from openai import OpenAI
from google.cloud import storage
import pathlib

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

openai.api_key = OPENAI_API_KEY
speech_client = speech.SpeechClient()

client = OpenAI(api_key=OPENAI_API_KEY)

async def convert_audio_to_wav(audio_path: str, file_type: str, message_id: str) -> tuple[str, float]:
    try:
        if file_type == "voice":
            audio = AudioSegment.from_ogg(audio_path)
        elif file_type == "mp3":
            audio = AudioSegment.from_mp3(audio_path)
        elif file_type == "wav":
            audio = AudioSegment.from_wav(audio_path)
        else:
            raise ValueError("Unsupported audio format. Use MP3 or WAV.")
        
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        wav_path = f"converted_{message_id}.wav"
        audio.export(wav_path, format="wav")
        duration_seconds = len(audio) / 1000.0
        return wav_path, duration_seconds
    except Exception as e:
        raise RuntimeError(f"Failed to convert audio: {str(e)}")

async def recognize_audio(wav_path: str, duration_seconds: float, lang_code: str) -> str:
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=lang_code,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
    )
    
    if duration_seconds > 60:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob_name = f"audio-{os.path.basename(wav_path)}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(wav_path)
        gcs_uri = f"gs://{GCS_BUCKET_NAME}/{blob_name}"
        audio_data = speech.RecognitionAudio(uri=gcs_uri)
        operation = speech_client.long_running_recognize(config=config, audio=audio_data)
        response = operation.result(timeout=300)
        blob.delete()
    else:
        with open(wav_path, "rb") as audio_file:
            content = audio_file.read()
        audio_data = speech.RecognitionAudio(content=content)
        response = speech_client.recognize(config=config, audio=audio_data)
    
    if response.results:
        return " ".join(result.alternatives[0].transcript for result in response.results)
    return ""

def cleanup_files(audio_path: str, wav_path: str, duration_seconds: float = 0) -> None:
    if os.path.exists(audio_path):
        os.remove(audio_path)
    if os.path.exists(wav_path):
        os.remove(wav_path)

async def send_text_with_buttons(update: Update, text: str, button_type: str) -> None:
    if button_type == "process_gpt":
        keyboard = [[InlineKeyboardButton("Process with GPT 📝", callback_data="process_gpt")]]
    elif button_type == "get_summary":
        keyboard = [[InlineKeyboardButton("Get Summary 📋", callback_data="get_summary")]]
    else:
        keyboard = []
    
    reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
    await update.message.reply_text(f"Recognized text 📜:\n{text}", reply_markup=reply_markup)

async def segment_text(text: str, max_chunk_size: int = 1000) -> list[str]:
    """Сегментує текст на частини для обробки GPT."""
    words = text.split()
    return [" ".join(words[i:i + max_chunk_size]) for i in range(0, len(words), max_chunk_size)]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hello! Send me a voice message or an audio file (MP3, WAV), and I'll convert it to text 🎙️\n"
        "To change the recognition language, use the /language command 🌐"
    )

async def choose_language(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton("Ukrainian 🇺🇦", callback_data="lang_uk-UA"),
            InlineKeyboardButton("English 🇬🇧", callback_data="lang_en-US"),
        ],
        [
            InlineKeyboardButton("German 🇩🇪", callback_data="lang_de-DE"),
            InlineKeyboardButton("French 🇫🇷", callback_data="lang_fr-FR"),
        ],
        [
            InlineKeyboardButton("Spanish 🇪🇸", callback_data="lang_es-ES"),
            InlineKeyboardButton("Japanese 🇯🇵", callback_data="lang_ja-JP"),
        ],
        [
            InlineKeyboardButton("Italian 🇮🇹", callback_data="lang_it-IT"),
            InlineKeyboardButton("Polish 🇵🇱", callback_data="lang_pl-PL"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Choose recognition language 🌍:", reply_markup=reply_markup)

async def handle_language_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    lang_code = query.data.replace("lang_", "")
    context.user_data["lang_code"] = lang_code
    await query.edit_message_text(f"Recognition language set: {lang_code} ✅")
    await query.message.reply_text("Waiting for your audio message or file 🎙️")

async def punctuate_text_with_gpt(text: str, lang_code: str) -> str:
    try:
        system_prompt = "You are an assistant that adds and corrects punctuation in text to make it grammatically correct and natural in the user's language."
        user_prompt = f"Add punctuation to this text and structure sentences grammatically correctly ({lang_code}):\n{text}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=4000,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return text

async def summarize_text_with_gpt(text: str, lang_code: str) -> str:
    try:
        system_prompt = "You are an assistant that creates a short summary of text, preserving the main ideas, in the user's language."
        user_prompt = f"Create a short summary (up to 100 words) of this text, keeping the language ({lang_code}):\n{text}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Failed to create summary"

async def handle_summary_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    lang_code = context.user_data.get("lang_code", "uk-UA")
    formatted_text = context.user_data.get("last_formatted_text", "")
    
    if not formatted_text:
        await query.edit_message_text("No text found for summary 😔")
        return
    
    summary = await summarize_text_with_gpt(formatted_text, lang_code)
    await query.edit_message_text(f"Short summary 📝:\n{summary}")

async def handle_gpt_processing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    lang_code = context.user_data.get("lang_code", "uk-UA")
    raw_text = context.user_data.get("last_raw_text", "")
    
    if not raw_text:
        await query.edit_message_text("No text found for processing 😔")
        return
    
    chunks = await segment_text(raw_text)
    formatted_text = ""
    for chunk in chunks:
        chunk_formatted = await punctuate_text_with_gpt(chunk, lang_code)
        formatted_text += chunk_formatted + " "
    
    context.user_data["last_formatted_text"] = formatted_text.strip()
    
    if len(formatted_text) > 500:
        keyboard = [[InlineKeyboardButton("Get Summary 📋", callback_data="get_summary")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(f"Processed text 📜:\n{formatted_text.strip()}", reply_markup=reply_markup)
    else:
        await query.edit_message_text(f"Processed text 📜:\n{formatted_text.strip()}")

async def process_audio(audio_path: str, file_type: str, message_id: str, context: ContextTypes.DEFAULT_TYPE, update: Update) -> None:
    try:
        wav_path, duration_seconds = await convert_audio_to_wav(audio_path, file_type, message_id)
        
        await update.message.reply_text("Processing audio, please wait... ⏳")

        lang_code = context.user_data.get("lang_code", "uk-UA")
        raw_text = await recognize_audio(wav_path, duration_seconds, lang_code)
        
        if raw_text:
            context.user_data["last_raw_text"] = raw_text
            await send_text_with_buttons(update, raw_text, "process_gpt")
        else:
            await update.message.reply_text("Couldn't recognize the text 😔")

        cleanup_files(audio_path, wav_path, duration_seconds)
    
    except ValueError as e:
        await update.message.reply_text(str(e))
        cleanup_files(audio_path, wav_path if 'wav_path' in locals() else "")
    except Exception as e:
        await update.message.reply_text(f"Error processing audio 😔: {str(e)}")
        cleanup_files(audio_path, wav_path if 'wav_path' in locals() else "")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        voice_file = await update.message.voice.get_file()
        voice_path = f"voice_{update.message.message_id}.ogg"
        await voice_file.download_to_drive(voice_path)
        await process_audio(voice_path, "voice", str(update.message.message_id), context, update)
    except Exception as e:
        await update.message.reply_text(f"Error processing voice message 😔: {str(e)}")

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        audio_file = await update.message.audio.get_file()
        mime_type = update.message.audio.mime_type
        message_id = str(update.message.message_id)

        if mime_type == "audio/mpeg":
            file_type = "mp3"
        elif mime_type == "audio/wav":
            file_type = "wav"
        else:
            await update.message.reply_text("Unsupported audio format 😔. Please use MP3 or WAV.")
            return

        audio_path = f"audio_{message_id}.{file_type}"
        await audio_file.download_to_drive(audio_path)
        await process_audio(audio_path, file_type, message_id, context, update)
    except Exception as e:
        await update.message.reply_text(f"Error processing audio file 😔: {str(e)}")

async def handle_audio_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        document = update.message.document
        mime_type = document.mime_type
        file_name = document.file_name or "audio"
        ext = pathlib.Path(file_name).suffix.lower()
        file_type = None

        if ext in [".wav", ".wave"]:
            file_type = "wav"
        elif ext in [".mp3"]:
            file_type = "mp3"
        else:
            await update.message.reply_text(f"Unsupported document format 😔. Extension: {ext}")
            return

        file = await document.get_file()
        message_id = str(update.message.message_id)
        audio_path = f"document_audio_{message_id}{ext}"
        await file.download_to_drive(audio_path)

        await process_audio(audio_path, file_type, message_id, context, update)
    except Exception as e:
        await update.message.reply_text(f"Error processing document audio 😔: {str(e)}")

def main() -> None:
    try:
        application = Application.builder().token(TELEGRAM_TOKEN).build()
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("language", choose_language))
        application.add_handler(CallbackQueryHandler(handle_language_choice, pattern="lang_"))
        application.add_handler(CallbackQueryHandler(handle_summary_button, pattern="get_summary"))
        application.add_handler(CallbackQueryHandler(handle_gpt_processing, pattern="process_gpt"))
        application.add_handler(MessageHandler(filters.VOICE, handle_voice))
        application.add_handler(MessageHandler(filters.AUDIO, handle_audio))
        application.add_handler(MessageHandler(filters.Document.AUDIO, handle_audio_document))
        application.run_polling()
    except KeyboardInterrupt:
        pass
    except Exception:
        pass

if __name__ == "__main__":
    main()