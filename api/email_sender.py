import os
import smtplib
import ssl
from config.app_config import Email_App_Password, Email_Username, Email_Recievers
from email.message import EmailMessage

# Define email sender and receiver
email_sender = Email_Username
email_password = Email_App_Password
email_receiver = Email_Recievers

# Set the subject and body of the email
subject = 'Safety Alert'
body = """
This is an alert
"""

em = EmailMessage()
em['From'] = email_sender
em['To'] = email_receiver
em['Subject'] = subject
em.set_content(body)

# Add SSL (layer of security)
context = ssl.create_default_context()

# Log in and send the email
with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
    smtp.login(email_sender, email_password)
    smtp.sendmail(email_sender, email_receiver, em.as_string())


def send_email(to_email, subject, body):
    em = EmailMessage()
    em['From'] = Email_Username
    em['To'] = to_email
    em['Subject'] = subject
    em.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(Email_Username, Email_App_Password)
        smtp.send_message(em)
