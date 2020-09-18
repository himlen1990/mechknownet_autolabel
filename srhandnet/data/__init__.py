import os.path as osp


data_dir = osp.abspath(osp.dirname(__file__))


def font_path():
    return osp.join(data_dir, 'Roboto-Bold.ttf')
