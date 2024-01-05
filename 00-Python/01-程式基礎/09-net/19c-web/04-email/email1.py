import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def email(user, password, title, body):
    #The mail addresses and password
    sender_address = f'{user}@gmail.com'
    sender_pass = password
    receiver_address = 'ccc@nqu.edu.tw'
    #Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = title
    #The body and the attachments for the mail
    message.attach(MIMEText(body, 'plain'))
    #Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
    session.starttls() #enable security
    session.login(sender_address, sender_pass) #login with mail_id and password
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()

email('user', 'password', 'python sent email', 'hello!')
print('Mail Sent')
