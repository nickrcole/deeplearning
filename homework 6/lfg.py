import requests
import random
import schedule
import time

messages = ["Push yourself beyond the limits; pain is temporary, pride is forever.",
"Run like your critics are cheering for you to fail. Show them the strength of your will.",
"Embrace the challenge, embrace the burn. This is where champions are made.",
"Your body can withstand anything. It's your mind you need to convince. Convince it.",
"Every step you take is a step toward a stronger, faster, better version of yourself.",
"Run harder than yesterday if you want a different tomorrow.",
"Turn your can'ts into cans and your dreams into plans. Run with purpose.",
"The pain you feel today will be the strength you feel tomorrow. Keep pushing!",
"Don't count the miles, make the miles count. Run with heart and determination.",
"You're not just running. You're building resilience, strength, and an unbreakable spirit.",
"Stop being a fucking bitch"]

def send():
    url = 'http://193.122.157.67:8000/nick_alerts'
    data = random.choice(messages)
    response = requests.post(url, data=data)

schedule.every(5).minutes.do(send)

while True:
    schedule.run_pending()
    time.sleep(1)