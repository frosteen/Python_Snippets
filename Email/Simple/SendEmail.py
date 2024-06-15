import smtplib


def sendemail(FROM, TO, SUBJECT, TEXT, EMAIL, PASSWORD):
    message = """
From: %s
To: %s
Subject: %s

%s
	""" % (
        FROM,
        ", ".join(TO),
        SUBJECT,
        TEXT,
    )
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(EMAIL, PASSWORD)
    server.sendmail(FROM, TO, message)
    server.quit()
