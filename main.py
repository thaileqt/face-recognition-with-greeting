import json

from utils.face_capture import face_capture
import os
from utils.generate_speech import generate_speech
from unidecode import unidecode


class Pipeline:
    def __init__(self, name: str):
        no_accent_name = unidecode(name)
        # append no_accent_name: name to name_config.json
        with open('name_config.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        data[no_accent_name] = name
        with open('name_config.json', 'w', encoding='utf-8') as f:
            json.dump(data, f)
        self.name = no_accent_name
        self.face_capture()
        self.train()
        self.generate_speech()

    def face_capture(self):
        print('Capturing faces for training...')
        face_capture(self.name)

    def train(self):
        print('Training model...')
        os.system('python3 utils/train.py')

    def generate_speech(self):
        print('Generating greeting speech...')
        generate_speech(self.name)


if __name__ == '__main__':
    name = input('Enter your name: ')
    Pipeline(name)