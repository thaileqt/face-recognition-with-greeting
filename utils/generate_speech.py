import json
import os


def generate_speech(text):
    with open('text.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
        os.chdir('vietTTS')
        origin_text = config[text]
        filename = text.strip().replace(' ', '')
        os.system(
            f'python3 -m vietTTS.synthesizer --lexicon-file assets/infore/lexicon.txt --text="{origin_text}" --output=../speechs/{filename}.wav --silence-duration 0.2')
        os.chdir('..')
