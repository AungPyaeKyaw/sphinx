import logging

debug = False
info = True
only = True
FORMAT = '%(asctime)-15s %(level)'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('sphinx')


def e(msg):
    if not debug:
        print('%s %s' % ('[Error]', msg))


def i(msg, this_only=False):
    if only and this_only:
        print('%s %s' % ('[Info]', msg))
    elif not only:
        print('%s %s' % ('[Info]', msg))


def w(msg):
    if not debug:
        print('%s %s' % ('[Warn]', msg))


def d(msg, this_only=False):
    if only and this_only:
        print('%s %s' % ('[Debug]', msg))
    elif not only and debug:
        print('%s %s' % ('[Debug]', msg))
