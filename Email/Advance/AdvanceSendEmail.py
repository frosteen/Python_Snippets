import mimetypes
import os
import smtplib
import ssl
from email.message import EmailMessage


def send_email(gmail_email, gmail_password, recipient, subject, body, file_path=None):
    """
    This function accepts gmail email and password then the
    email of recipient, email subject, email content and email attachment
    then sends it to the recipient.

    You must allow less secure apps in GMail for this to work.

    Return True on success, False otherwise.
    """

    # Create an EmailMessage object along with the parameters needed
    # to send an email
    message = EmailMessage()
    message["From"] = gmail_email
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)

    # MIME type and sub type define what type of file is
    # going to be attached to it can be included in email
    # since email sends the email itself as a series of characters
    # and neede to be serialized and deserialized.
    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type, mime_subtype = mime_type.split("/", 1)

    # print(mime_type, mime_subtype)

    # Add the conents of the file by reading it as binary mode.
    with open(file_path, "rb") as f:
        message.add_attachment(
            f.read(),
            maintype=mime_type,
            subtype=mime_subtype,
            filename=os.path.basename(file_path),
        )

    # print(message)

    message = message.as_string()
    context = ssl.create_default_context()

    # Create and SMTP email object based on Gmail.
    # SMTP SSL uses port 465
    mail_server = smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context)
    print(mail_server.login(gmail_email, gmail_password))
    mail_server.sendmail(gmail_email, recipient, message)

    mail_server.quit()

    return True


if __name__ == "__main__":
    import getpass

    gmail_email = "unmanned.vehicle.2022@gmail.com"
    gmail_password = "unmannedvehicle@2022"

    recipient = "unmanned.vehicle.2022@gmail.com"
    subject = "Python Email Test"
    body = "I'm learning to send email using Python with email.message module."
    file_path = "test_file.txt"

    send_email(gmail_email, gmail_password, recipient, subject, body, file_path)
