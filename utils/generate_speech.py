import json
import os


def generate_speech(name):
    with open('name_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
        os.chdir('vietTTS')
        text = "chào bạn " + config[name]
        os.system(
            f'python3 -m vietTTS.synthesizer --lexicon-file assets/infore/lexicon.txt --text="{text}" --output=../speechs/{name}.wav --silence-duration 0.2')
        os.chdir('..')
