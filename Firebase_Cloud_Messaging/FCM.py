from firebase_admin import messaging


def send_notification(title, body):
    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body,
        ),
        topic="immediate-assistance",
    )
    response = messaging.send(message)
    print("Successfully sent message:", response)


if __name__ == "__main__":
    import firebase_admin
    from firebase_admin import credentials

    cred = credentials.Certificate("voice-recognition-device-firebase-adminsdk.json")
    firebase_admin.initialize_app(cred)

    send_notification("PI TEST", "PI TEST")
