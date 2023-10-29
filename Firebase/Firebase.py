import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


cred = credentials.Certificate(
    "guide-blind-people-firebase-adminsdk-5ol9c-a66b782e74.json"
)
firebase_admin.initialize_app(
    cred, {"databaseURL": "https://guide-blind-people-default-rtdb.firebaseio.com/"}
)


def update(location="/", json_query={}):
    db.reference(location).update(json_query)


def get(location="/"):
    return db.reference(location).get()


def delete(location="/"):
    db.reference(location).delete()


if __name__ == "__main__":
    update(
        "/",
        {
            "Number": 2,
            "isWorking": False,
            "Sentence": "Hello world.",
        },
    )
    print(get("/"))
