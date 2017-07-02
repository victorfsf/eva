import glob
import os
import requests
import logging

MODELS_PATH = os.path.join(
    os.path.expanduser('~'),
    '.eva-models'
)


def set_models_path(path):
    if path:
        global MODELS_PATH
        MODELS_PATH = path


def download(path=None):
    set_models_path(path)

    logger = logging.getLogger(__name__)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)

    def from_url(url):
        raw = requests.get(url)
        if not os.path.isdir(MODELS_PATH):
            os.mkdir(MODELS_PATH)
        if raw.status_code != 200:
            logger.info('Model URL not found: %s' % url)
            return
        model_name = os.path.join(
            MODELS_PATH, os.path.basename(url).split('?')[0]
        )
        with open(model_name, 'wb') as f:
            f.write(raw.content)

    for model in glob.glob('models/*'):
        name = os.path.basename(model)
        logger.info('Downloading model: %s' % name)
        from_url(
            'https://github.com/victorfsf/eva/'
            'blob/master/models/%s?raw=true' % name
        )
