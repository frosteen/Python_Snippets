import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone(device_index=0) as source:

    try:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        text = r.recognize_google(audio)
        print(text)

    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        print("unknown error occured")
