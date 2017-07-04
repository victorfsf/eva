import glob
import os
import requests
import logging

EVA_PATH = os.path.join(
    os.path.expanduser('~'),
    '.eva',
)


def set_eva_path(path):
    if path:
        global EVA_PATH
        EVA_PATH = path


def download(path=None):
    set_eva_path(path)

    logger = logging.getLogger(__name__)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)

    def from_url(url, subfolder):
        raw = requests.get(url)
        folder = os.path.join(EVA_PATH, subfolder)
        if not os.path.isdir(EVA_PATH):
            os.mkdir(EVA_PATH)
        if not os.path.isdir(folder):
            os.mkdir(folder)
        if raw.status_code != 200:
            logger.info('URL not found: %s' % url)
            return
        filename = os.path.join(
            folder, os.path.basename(url).split('?')[0]
        )
        with open(filename, 'wb') as f:
            f.write(raw.content)

    for model in glob.glob('models/*'):
        name = os.path.basename(model)
        logger.info('Downloading: %s' % name)
        from_url(
            'https://github.com/victorfsf/eva/'
            'blob/master/models/%s?raw=true' % name,
            subfolder='models'
        )
