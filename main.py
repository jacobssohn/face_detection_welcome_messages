import face_recognition
import os, sys, time, random
import cv2
import numpy as np
import math

import openai.error
import playsound
# from chatgpt_wrapper import ChatGPT
from chatGPT_API import GPTBot
from gtts import gTTS


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2)

    if face_distance > face_match_threshold:
        return str(round(linear_val*100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

class FaceRecognition:

    face_locations = []
    face_encodings = []
    face_names = []
    sounds = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    sound_timeout = False
    timeout_timer = 0


    def __init__(self):
        self.encode_faces()
        self.encode_sounds()
        self.bot = GPTBot()

    def encode_faces(self):

        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)

        print(self.known_face_names)

    def encode_sounds(self):

        for sound in os.listdir('sounds'):
            self.sounds.append(f'sounds/{sound}')

        print(self.sounds)

    def generate_prompt(self, person):
        wordlist = ['welcome', 'sexy', 'bad', 'cunt', 'twat', 'cheerful', 'puzzled']
        words = [wordlist[random.randint(1, len(wordlist)-1)] for _ in range(5)]
        # words.append(person)
        prompt = f'Write a welcome message for {person} using these words: '
        for word in words:
            prompt += word
            prompt += ', '

        return prompt

    def generate_text(self, prompt):
        try:
            return self.bot.generate_response(prompt)
        except openai.error.RateLimitError:
            return 0

    def play_audio(self, message):
        audio = gTTS(text=message, lang='en', slow=False, tld='us')
        audio.save('sounds/example.mp3')
        playsound.playsound('sounds/example.mp3')
        os.remove('sounds/example.mp3')

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found. ')

        while True:
            ret, frame = video_capture.read()

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]

                # find all faces in current frame
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'Unknown'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if not self.sound_timeout:
                        self.timeout_timer = time.time()
                        self.sound_timeout = True

                        # playsound.playsound(self.sounds[best_match_index], block=False)

                        person = self.known_face_names[best_match_index].split('.')[0]

                        message = self.generate_text(self.generate_prompt(person))
                        if message:
                            print(message)
                            self.play_audio(message)

                    elif time.time() - self.timeout_timer > 10:
                        self.sound_timeout = False

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame

            # display annotations
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom + 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    # def play_sound(self):



if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
    # bot = GPTBot()
    # response = bot.generate_response('U gay boi')
    # audio = gTTS(text=response, lang='es', slow=False, tld='us')
    # audio.save('sounds/example.mp3')
    # playsound.playsound('sounds/example.mp3')
    # print(response)