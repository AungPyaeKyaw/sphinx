import logging

debug = False
info = True
FORMAT = '%(asctime)-15s %(level)'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('sphinx')


def e(msg):
    if not debug:
        print('%s %s' % ('[Error]', msg))


def i(msg):
    if info:
        print('%s %s' % ('[Info]', msg))


def w(msg):
    if not debug:
        print('%s %s' % ('[Warn]', msg))


def d(msg):
    if debug:
        print('%s %s' % ('[Debug]', msg))
