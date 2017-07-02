from decouple import config
from eva.api.telegram.interface import TelegramAPI
from flask import Flask
from flask import request
import json

app = Flask('eva')
bot = TelegramAPI(config('TELEGRAM_API_KEY'))


@app.route('/', methods=['POST'])
def webhook():
    post = json.loads(request.data)
    message = post['message']
    chat_id = message['chat']['id']
    user_name = message['from']['first_name']
    if message['text']:
        return json.dumps(
            bot.send_message(chat_id, 'Olá %s' % user_name)
        )
    return json.dumps({
        'ok': False
    })
