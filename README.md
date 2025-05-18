# Telegram-Voice-Bot
This is a  Telegram bot built with Python that can:

-- Receive voice messages and audio files (MP3, WAV, OGG, even as documents)

-- Automatically convert and transcribe audio using Google Speech-to-Text API

-- Format and punctuate the transcribed text using OpenAI GPT (gpt-3.5-turbo)

-- Optionally summarize long transcriptions

-- Support multiple languages (English, Ukrainian, German, etc.)

Features
-- Supports audio via voice, audio, and document messages

-- Converts any common audio format to standardized WAV

-- Uploads long files to Google Cloud Storage for asynchronous recognition

-- Utilizes GPT for punctuation, grammar, and summaries

-- Includes in-chat buttons for processing and summarizing text

-- Built with python-telegram-bot, pydub, openai, and Google APIs
