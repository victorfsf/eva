import json
import requests


def validate_json(fn):
    """Validates a JSON string.

    Keyword arguments:
    fn -- a response object.
    """
    def wrapper(self, *args, **kwargs):
        try:
            return json.loads(fn(self, *args, **kwargs).content)
        except (ValueError, TypeError):
            return {
                "ok": False,
                "error_code": 400,
                "description": "[Error]: Invalid JSON Format"
            }
    return wrapper


class InvalidBotToken(Exception):
    pass


class TelegramAPI(object):
    """https://core.telegram.org/bots/api#available-methods"""

    def __init__(self, token, *args, **kwargs):
        self._token = token
        self._url = kwargs.pop('url', 'https://api.telegram.org/bot')
        response = self.get_me()
        if not response.get('ok'):
            raise InvalidBotToken(u'Invalid Telegram Bot Token: "%s"!' % token)
        self.bot_id = response.get('result').get('id')
        super(TelegramAPI, self).__init__(*args, **kwargs)

    def __repr__(self):
        return "TelegramAPI(bot_id={})".format(self.bot_id)

    @validate_json
    def _request(self, method, files=None, **kwargs):
        """Base method to execute the requests to the Telegram API
           with the given method name, files and arguments.

        Keyword arguments:
        method -- The method's name.
        files  -- The files list to be sent through multipart/form-data.
        """
        return requests.get(
            '{}{}/{}'.format(
                self._url, self._token, method
            ), data=kwargs, files=files
        )

    def _send_file(self, method, input_type, file_id, **kwargs):
        """Base method to send a file through multipart/form-data
           with the given method, file type, file path and keyword arguments.

        Keyword arguments:
        method     -- The method's name.
        input_type -- The type of the file.
                      "audio" or "voice" for audio files,
                      "document" for pdf files,
                      "photo" or "sticker" for image files,
                      "video" for video files.
        file_id   -- The file's local path.
        resend     -- If True, the file_id argument will be treated as
                      a Telegram's file_id. If not, it will be treated as
                      the path to a normal binary file.
        """
        if kwargs.pop('resend', False):
            kwargs[input_type] = file_id
            return self._request(method, **kwargs)
        return self._request(
            method, files={
                input_type: open(file_id, 'r')
            }, **kwargs
        )

    def download(self, file_id, save_to=None):
        """Base method to download a file through its file path.

        Keyword arguments:
        file_id -- The file's path on Telegram.
        save_to  -- The local path to where the file will be saved.
        """
        response = requests.get(
            '{}{}/{}'.format(
                self._url, self._token, file_id
            )
        ).content
        if save_to:
            with open(save_to, 'w') as f:
                f.write(response)
            return save_to
        return response

    def download_file(self, file_id, save_to=None):
        """Base method to download a file through its id and save it.

        Keyword arguments:
        file_id -- The file's Telegram ID.
        save_to -- The local path to where the file will be saved.
        """
        response = self.get_file(file_id=file_id)
        if response.get('ok'):
            file_id = response.get('result').get('file_path')
            return self.download(file_id, save_to)
        return response

    # Getting updates: https://core.telegram.org/bots/api#getting-updates

    def set_webhook(self, url, **kwargs):
        # https://core.telegram.org/bots/api#setwebhook
        return self._request('setWebhook', url=url, **kwargs)

    def delete_webhook(self):
        # https://core.telegram.org/bots/api#deletewebhook
        return self._request('deleteWebhook')

    def get_webhook_info(self):
        # https://core.telegram.org/bots/api#getwebhookinfo
        return self._request('getWebhookInfo')

    def get_updates(self, **kwargs):
        # https://core.telegram.org/bots/api#getupdates
        return self._request('getUpdates', **kwargs)

    # Base methods: https://core.telegram.org/bots/api#available-methods

    def get_me(self):
        # https://core.telegram.org/bots/api#getme
        return self._request('getMe')

    def send_message(self, chat_id, text, **kwargs):
        # https://core.telegram.org/bots/api#sendmessage
        reply_markup = kwargs.pop('reply_markup', None)
        if isinstance(reply_markup, dict):
            kwargs['reply_markup'] = json.dumps(reply_markup)
        return self._request(
            'sendMessage',  chat_id=chat_id, text=text, **kwargs
        )

    def foward_message(self, chat_id, from_chat_id, message_id, **kwargs):
        # https://core.telegram.org/bots/api#forwardmessage
        reply_markup = kwargs.pop('reply_markup', None)
        if isinstance(reply_markup, dict):
            kwargs['reply_markup'] = json.dumps(reply_markup)
        return self._request(
            'forwardMessage', chat_id=chat_id,
            from_chat_id=from_chat_id,
            message_id=message_id, **kwargs
        )

    def send_photo(self, chat_id, file_id, **kwargs):
        # https://core.telegram.org/bots/api#sendphoto
        return self._send_file(
            'sendPhoto', 'photo',
            file_id, chat_id=chat_id, **kwargs
        )

    def send_audio(self, chat_id, file_id, **kwargs):
        # https://core.telegram.org/bots/api#sendaudio
        return self._send_file(
            'sendAudio', 'audio',
            file_id, chat_id=chat_id, **kwargs
        )

    def send_document(self, chat_id, file_id, **kwargs):
        # https://core.telegram.org/bots/api#senddocument
        return self._send_file(
            'sendDocument', 'document',
            file_id, chat_id=chat_id, **kwargs
        )

    def send_sticker(self, chat_id, file_id, **kwargs):
        # https://core.telegram.org/bots/api#sendsticker
        return self._send_file(
            'sendSticker', 'sticker',
            file_id, chat_id=chat_id, **kwargs
        )

    def send_video(self, chat_id, file_id, **kwargs):
        # https://core.telegram.org/bots/api#sendvideo
        return self._send_file(
            'sendVideo', 'video',
            file_id, chat_id=chat_id, **kwargs
        )

    def send_voice(self, chat_id, file_id, **kwargs):
        # https://core.telegram.org/bots/api#sendvoice
        return self._send_file(
            'sendVoice', 'voice',
            file_id, chat_id=chat_id, **kwargs
        )

    def send_location(self, chat_id, latitude, longitude, **kwargs):
        # https://core.telegram.org/bots/api#sendlocation
        return self._request(
            'sendLocation', chat_id=chat_id,
            latitude=latitude,
            longitude=longitude, **kwargs
        )

    def send_venue(self, chat_id, latitude, longitude,
                   title, address, **kwargs):
        # https://core.telegram.org/bots/api#sendvenue
        return self._request(
            'sendVenue', chat_id=chat_id,
            latitude=latitude, longitude=longitude,
            title=title, address=address, **kwargs
        )

    def send_contact(self, chat_id, phone_number, first_name, **kwargs):
        # https://core.telegram.org/bots/api#sendcontact
        return self._request(
            'sendContact', chat_id=chat_id,
            phone_number=phone_number,
            first_name=first_name,
            **kwargs
        )

    def send_chat_action(self, chat_id, action):
        # https://core.telegram.org/bots/api#sendchataction
        return self._request('sendChatAction', chat_id=chat_id, action=action)

    def get_user_profile_photos(self, user_id, **kwargs):
        # https://core.telegram.org/bots/api#getuserprofilephotos
        return self._request(
            'getUserProfilePhotos',
            user_id=user_id, **kwargs
        )

    def get_file(self, file_id):
        # https://core.telegram.org/bots/api#getfile
        return self._request('getFile', file_id=file_id)

    def kick_chat_member(self, chat_id, user_id):
        # https://core.telegram.org/bots/api#kickchatmember
        return self._request(
            'kickChatMember', chat_id=chat_id, user_id=user_id
        )

    def leave_chat(self, chat_id):
        # https://core.telegram.org/bots/api#leavechat
        return self._request('leaveChat', chat_id=chat_id)

    def unban_chat_member(self, chat_id, user_id):
        # https://core.telegram.org/bots/api#unbanchatmember
        return self._request(
            'unbanChatMember', chat_id=chat_id, user_id=user_id
        )

    def get_chat(self, chat_id):
        # https://core.telegram.org/bots/api#getchat
        return self._request('getChat', chat_id=chat_id)

    def get_chat_administrators(self, chat_id):
        # https://core.telegram.org/bots/api#getchatadministrators
        return self._request('getChatAdministrators', chat_id=chat_id)

    def get_chat_members_count(self, chat_id):
        # https://core.telegram.org/bots/api#getchatmemberscount
        return self._request('getChatMembersCount', chat_id=chat_id)

    def get_chat_member(self, chat_id, user_id):
        # https://core.telegram.org/bots/api#getchatmember
        return self._request('getChatMember', chat_id=chat_id, user_id=user_id)

    def answer_callback_query(self, callback_query_id, **kwargs):
        # https://core.telegram.org/bots/api#answercallbackquery
        return self._request(
            'answerCallbackQuery',
            callback_query_id=callback_query_id, **kwargs
        )

    # Updating messages: https://core.telegram.org/bots/api#updating-messages

    def edit_message_text(self, text, **kwargs):
        # https://core.telegram.org/bots/api#editmessagetext
        return self._request('editMessageText', text=text, **kwargs)

    def edit_message_caption(self, **kwargs):
        # https://core.telegram.org/bots/api#editmessagecaption
        return self._request('editMessageCaption', **kwargs)

    def edit_message_reply_markup(self, **kwargs):
        # https://core.telegram.org/bots/api#editmessagereplymarkup
        return self._request('editMessageReplyMarkup', **kwargs)

    # Inline mode: https://core.telegram.org/bots/api#inline-mode

    def answer_inline_query(self, inline_query_id, results, **kwargs):
        # https://core.telegram.org/bots/api#answerinlinequery
        return self._request(
            'answerInlineQuery',
            inline_query_id=inline_query_id,
            results=results, **kwargs
        )

    # Games: https://core.telegram.org/bots/api#games

    def send_game(self, chat_id, game_short_name, **kwargs):
        # https://core.telegram.org/bots/api#sendgame
        return self._request(
            'sendGame', chat_id=chat_id, game_short_name=game_short_name,
            **kwargs
        )

    def set_game_score(self, user_id, score, **kwargs):
        # https://core.telegram.org/bots/api#setgamescore
        return self._request(
            'setGameScore',
            user_id=user_id, score=score,
            **kwargs
        )

    def get_game_high_scores(self, user_id, **kwargs):
        # https://core.telegram.org/bots/api#getgamehighscores
        return self._request(
            'getGameHighScores',
            user_id=user_id, **kwargs
        )
