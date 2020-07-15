import smtplib
from email.message import EmailMessage

carriers = {
    'att': '@mms.att.net',
    'tmobile': ' @tmomail.net',
    'verizon': '@vtext.com',
    'sprint': '@page.nextel.com'
}


def send(message, numbers, carrier_names):
    # Replace the number with your own, or consider using an argument\dict for multiple people.
    auth = ('crcv.messenger.bot@gmail.com', 'crcv.is.best!')

    try:
        # make sure the passed values are iterable
        if type(numbers) not in [list, tuple]:
            numbers = (numbers,)
            carrier_names = (carrier_names,)

        # Establish a secure session with gmail's outgoing SMTP server using your gmail account
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(auth[0], auth[1])

        msg = EmailMessage()
        msg['from'] = auth[0]
        msg['subject'] = ''
        msg.set_content(message)

        # send message to all specified numbers
        for number, carrier in zip(numbers, carrier_names):
            # create 'to' full address and set it
            to_number = '{}{}'.format(number, carriers[carrier])
            msg['to'] = to_number

            # send the email
            server.sendmail(auth[0], to_number, str(msg))

    except Exception as e:
        print("Error sending message: {}".format(str(e)))
