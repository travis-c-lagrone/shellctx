"""
#!/usr/bin/env python3
shellctx
--------

Shell Context Helper

Web:      https://github.com/serwy/shellctx

Author:   Roger D. Serwy
Date:     2020-06-26
License:  GNU GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html


"""

import argparse
import datetime
import functools
import inspect
import json
import os  # TODO refactor uses of module `os.path` to `pathlib`
import sys

from abc import ABC, abstractmethod

__version__ = '1.0.0-dev.0'

WINDOWS = sys.platform.startswith('win')  # TODO refactor `sys.platform` constant to `platform` module
ISATTY = sys.stdout.isatty()

# ANSI coloring
COLOR = {
    '': '\033[0m',  # reset
    'black': '\033[0;30m',
    'red': '\033[0;31m',
    'green': '\033[0;32m',
    'blue': '\033[0;94m',
    'yellow': '\033[0;33m',
}

if WINDOWS or not ISATTY:  # TODO test more elegantly if terminal colors supported on Windows
    COLOR = dict.fromkeys(COLOR.keys(), '')

STYLE = {
    '': COLOR[''],  # blank
    'key': COLOR['green'],
    'value': COLOR['blue'],
    'time': COLOR['red'],
    'command': COLOR['blue'],
    'context': COLOR['blue'],
}

ENV_CTX_HOME = os.environ.get('CTX_HOME', None)
ENV_CTX_NAME = os.environ.get('CTX_NAME', None)
ENV_CTX_VERBOSE = int(os.environ.get('CTX_VERBOSE', 0))

CTX_HOME = ENV_CTX_HOME or os.path.expanduser('~/.ctx')
CTX_NAME_FILE = os.path.join(CTX_HOME, '_name.txt')

DEFAULT_CTX_NAME = 'main'

if ENV_CTX_NAME:
    CTX_NAME = ENV_CTX_NAME
elif os.path.exists(CTX_NAME_FILE):
    with open(CTX_NAME_FILE, 'r') as _fid:
        CTX_NAME = _fid.read().strip()
else:
    CTX_NAME = DEFAULT_CTX_NAME

CTX_FILE = os.path.join(CTX_HOME, CTX_NAME + '.json')
LOG_FILE = os.path.join(CTX_HOME, CTX_NAME + '.log')


class LazyData(ABC):
    def __init__(self, file):
        self._file = file
        self.__data = None
        self._modified = False

    @abstractmethod
    def _get_default_data(self):
        pass

    def _load(self):
        if os.path.exists(self._file):
            with open(self._file, 'rb') as fid:
                data = fid.read()
            self.__data = json.loads(data.decode('utf8'))
        else:
            self.__data = self._get_default_data()

    def load(self):
        if self.__data is None:
            self._load()

    @property
    def _data(self):
        if self.__data is None:
            self._load()
        return self.__data

    def __contains__(self, item):
        return item in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    @property
    def modified(self):
        return self._modified

    def _save(self):
        assert self._data is not None
        with open(self._file, 'wb') as fid:
            fid.write(json.dumps(self._data, indent=4).encode('utf8'))

    def save(self):
        if self._modified:
            self._save()


class ContextData(LazyData):
    def _get_default_data(self):
        return {}

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value
        self._modified = True

    def __delitem__(self, key):
        del self._data[key]
        self._modified = True

    def clear(self):
        self._data.clear()
        self._modified = True

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def update(self, other):
        self._data.update(other)
        self._modified = True

    def values(self):
        return self._data.values()


class LogData(LazyData):
    def _get_default_data(self):
        return []

    def append(self, timestamp, command, *args):
        self._data.append((timestamp, command, *args))


CTX = ContextData(CTX_FILE)
LOG = LogData(LOG_FILE)


print_err = functools.partial(print, file=sys.stderr)


def _print_version():
    s = ('shellctx version ',
         COLOR['red'],
         __version__,
         COLOR['']
         )
    print_err(''.join(s))


def _print_verbose():
    print_err('CTX_VERBOSE=%i' % ENV_CTX_VERBOSE)
    _print_version()
    if ENV_CTX_HOME:
        print_err('CTX_HOME=%s' % ENV_CTX_HOME)
    if ENV_CTX_NAME:
        print_err('CTX_NAME=%s' % ENV_CTX_NAME)


def _print_parsed_args(args):
    print_err('parsed args:')
    args_D = vars(args)
    max_key_len = max(len(k) for k in args_D.keys())
    for (key, value) in args_D.items():
        if key.startswith('_'):
            continue
        s = ('    ',
             STYLE['key'],
             key,
             STYLE[''],
             ':',
             ' ' * (1 + (max_key_len - len(key))),
             STYLE['value'],
             repr(value),
             STYLE['']
            )
        print_err(''.join(s))


def _print_debug(args):
    print_err('CTX_VERBOSE=%i' % ENV_CTX_VERBOSE)
    _print_version()
    print_err('CTX_HOME=%s' % ENV_CTX_HOME)
    print_err('CTX_NAME=%s' % ENV_CTX_NAME)
    print_err('context home: %s' % CTX_HOME)
    print_err('context file: %s' % CTX_FILE)
    _print_parsed_args(args)


def _print_full_items():
    # timestamp, key, value
    everything = [(v[0], k, v[1]) for k, v in CTX.items()]
    x = sorted(everything, reverse=True)
    s = ('Using context ', STYLE['context'], CTX_NAME, COLOR[''], '')
    if ENV_CTX_NAME:
        s = s + (' (set by CTX_NAME)', )
    if ENV_CTX_HOME:
        s = s + ((' (from CTX_HOME=%s)' % ENV_CTX_HOME),)
    print(''.join(s))
    s = ('There are ', STYLE['value'], str(len(everything)),
        COLOR[''], ' entries.\n')
    print(''.join(s))

    for ctime, _key, _value in x:
        s = (STYLE['time'],
             ctime, '    ',
             STYLE['key'], _key,
             COLOR[''], ' = ',
             STYLE['value'], str(_value),
             COLOR['']
             )
        print(''.join(s))


def get_now():
    return datetime.datetime.now().isoformat()


NOW = get_now()


def set_handler(parser, handler):
    parser.set_defaults(_handler=handler)


def handle(args):
    args._handler(args)


def handles(parser):
    """Decorates a function:
    - wraps it to accept a single parameter 'args',
    - sets the wrapped function as the default `func` of {parser}
    - returns the unwrapped function
    """
    def set_wrapped_handler(handler):
        keys = inspect.getfullargspec(handler).args
        has__args__ = '__args__' in keys
        if has__args__:
            keys = (k for k in keys if k != '__args__')
        def wrapped(args):
            kwargs = {k: getattr(args, k) for k in keys}
            if has__args__:
                kwargs['__args__'] = args
            return handler(**kwargs)
        set_handler(parser, wrapped)
        return handler
    return set_wrapped_handler


# TODO pass `help` for each parser argument
# TODO pass `help` (brief) for each parser/subparser
# TODO pass `description` (full) for each parser/subparser


parser = argparse.ArgumentParser(prog='ctx', allow_abbrev=False)
parser.add_argument('--version', action='store_true')
@handles(parser)
def handle_(version):
    if version:
        print(__version__)
    else:
        _print_full_items()


subparsers = parser.add_subparsers(title='commands')
type(subparsers).__getitem__ = lambda self, key: self._name_parser_map[key]
type(subparsers).keys = lambda self: self._name_parser_map.keys()


subparser = subparsers.add_parser('get')
subparser.add_argument('key')
@handles(subparser)
def handle_get(key):
    v = CTX[key][1]
    s = (STYLE['value'],
        v,
        COLOR[''],
        )
    print(''.join(s))


# TODO add `add` command (like set, except it errors out if the key already exists)
# TODO add 'replace' command (like set, except it errors out if the key does not already exist)
subparser = subparsers.add_parser('set')
subparser.add_argument('key')
subparser.add_argument('value')
subparser.add_argument('-e', '--entry', action='store_true')
subparser.add_argument('-p', '--path', action='store_true')
subparser.add_argument('-v', '--verbose', action='store_true', default=ENV_CTX_VERBOSE)
@handles(subparser)
def handle_set(key, value, entry, path, verbose):
    if path:
        base = os.getcwd()
        value = base if value == '.' else os.path.join(base, value)

    if entry:
        prefix = key + '_'
        N = len(prefix)
        keys = list(CTX.keys())
        suffix = [k[N:] for k in keys if k.startswith(prefix)]

        max_num = 0
        for num in suffix:
            try:
                num = int(num)
            except:
                pass
            else:
                max_num = max(num, max_num)
        next_num = max_num + 1

        key = prefix + ('%03i' % next_num)
        assert(key not in CTX)

    LOG.append(NOW, 'set', key, value)
    CTX[key] = (NOW, value)

    if verbose:
        print(''.join((
            STYLE['key'],
            key,
            COLOR[''],
            '=',
            STYLE['value'],
            value,
            COLOR['']
        )))


# TODO add `force` option to `del` command
subparser = subparsers.add_parser('del')
subparser.add_argument('keys', nargs='+', metavar='key')
subparser.add_argument('-p', '--pop', action='store_true')
subparser.add_argument('-v', '--verbose', action='store_true', default=ENV_CTX_VERBOSE)
@handles(subparser)
def handle_del(keys, pop, verbose):
    for key in keys:
        if key not in CTX:
            raise KeyError(key)

    popped_keys = set()
    for key in keys:
        if key in popped_keys:
            continue

        value = CTX[key][1]

        LOG.append(NOW, 'del', key, value)
        del CTX[key]

        if pop:
            print(value)

        if verbose:
            print(''.join((
                STYLE['key'],
                key,
                STYLE[''],
                '=',
            )))

        popped_keys.add(key)


subparser = subparsers.add_parser('copy')
subparser.add_argument('old-key')
subparser.add_argument('new-key')
@handles(subparser)
def handle_copy(old_key, new_key):
    LOG.append(NOW, 'copy', old_key, new_key)
    CTX[new_key] = (NOW, CTX[old_key][1])


subparser = subparsers.add_parser('rename')
subparser.add_argument('old-key')
subparser.add_argument('new-key')
@handles(subparser)
def handle_rename(old_key, new_key):
    LOG.append(NOW, 'rename', old_key, new_key)
    CTX[new_key] = CTX[old_key]
    del CTX[old_key]


def _get_contexts(include_default=True):
    files = os.listdir(CTX_HOME)
    ext = '.json'
    names = [f.rpartition(ext)[0] for f in files if f.endswith(ext)]
    if CTX_NAME not in names:
        names.append(CTX_NAME)
    if not include_default:
        try:
            names.remove(DEFAULT_CTX_NAME)
        except:
            pass
    names.sort()
    return names


subparser = subparsers.add_parser('get-ctx')
subparser.add_argument('-a', '--all', action='store_true')
@handles(subparser)
def handle_get_ctx(all):
    if all:
        for name in _get_contexts():
            print(name)
    else:
        print(CTX_NAME)


def _set_ctx(name):
    with open(CTX_NAME_FILE, 'w') as fid:
        fid.write(name)


subparser = subparsers.add_parser('set-ctx')
subparser.add_argument('name', nargs='?', default=DEFAULT_CTX_NAME)
subparser.add_argument('-v', '--verbose', action='store_true', default=ENV_CTX_VERBOSE)
@handles(subparser)
def handle_set_ctx(name, verbose):
    if ENV_CTX_NAME and name != ENV_CTX_NAME:
        print_err(''.join((
            COLOR['red'],
            'context set by CTX_NAME as ',
            STYLE['context'],
            ENV_CTX_NAME,
            COLOR['red'],
            '. Not switching.',
        )))
        return 1

    if name != CTX_NAME:
        _set_ctx(name)

    if verbose:
        print(''.join(('switching to "',
            STYLE['context'],
            name,
            COLOR[''],
            '" from "',
            STYLE['context'],
            CTX_NAME,
            COLOR[''],
            '"',
        )))


subparser = subparsers.add_parser('del-ctx')
subparser.add_argument('name', choices=_get_contexts(include_default=False))
@handles(subparser)
def handle_del_ctx(name):
    ctx_file = os.path.join(CTX_HOME, name + '.json')
    if os.path.exists(ctx_file):
        os.remove(ctx_file)

    log_file = os.path.join(CTX_HOME, name + '.log')
    if os.path.exists(log_file):
        os.remove(log_file)

    if name == CTX_NAME:
        _set_ctx(DEFAULT_CTX_NAME)


subparser = subparsers.add_parser('copy-ctx')
subparser.add_argument('old-name')
subparser.add_argument('new-name')
subparser.add_argument('--force', action='store_true')
@handles(subparser)
def handle_copy_ctx(old_name, new_name, force):
    old_ctx_file = os.path.join(CTX_HOME, old_name + '.json')
    old_log_file = os.path.join(CTX_HOME, old_name + '.log')

    new_ctx_file = os.path.join(CTX_HOME, new_name + '.json')
    new_log_file = os.path.join(CTX_HOME, new_name + '.log')

    if not force:
        if os.path.exists(new_ctx_file):
            raise FileExistsError(new_ctx_file)
        if os.path.exists(new_log_file):
            raise FileExistsError(new_log_file)

    import shutil
    shutil.copy(old_ctx_file, new_ctx_file)
    shutil.copy(old_log_file, new_log_file)


subparser = subparsers.add_parser('rename-ctx')
subparser.add_argument('old-name')
subparser.add_argument('new-name')
subparser.add_argument('--force', action='store_true')
@handles(subparser)
def handle_rename_ctx(old_name, new_name, force):
    old_ctx_file = os.path.join(CTX_HOME, old_name + '.json')
    old_log_file = os.path.join(CTX_HOME, old_name + '.log')

    new_ctx_file = os.path.join(CTX_HOME, new_name + '.json')
    new_log_file = os.path.join(CTX_HOME, new_name + '.log')

    if os.path.exists(new_ctx_file):
        if force:
            os.remove(new_ctx_file)
        else:
            raise FileExistsError(new_ctx_file)
    if os.path.exists(new_log_file):
        if force:
            os.remove(new_log_file)
        else:
            raise FileExistsError(new_log_file)

    os.rename(old_ctx_file, new_ctx_file)
    os.rename(old_log_file, new_log_file)


subparser = subparsers.add_parser('shell')
subparser.add_argument('cmd-key', metavar='command-key')
subparser.add_argument('arg-keys', nargs='*', metavar='argument-keys')
subparser.add_argument('-d', '--dry-run', action='store_true')
subparser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction, default=bool(ENV_CTX_VERBOSE))
@handles(subparser)
def handle_shell(cmd_key, arg_keys, dry_run, verbose):
    cmd = CTX[cmd_key][1]
    args_ = [CTX[k][1] for k in arg_keys]
    sh_cmd = f"{cmd} {' '.join(args_)}" if args_ else cmd

    s = ('shell command: ',
        STYLE['command'],
        sh_cmd,
        COLOR[''],
        )
    if verbose:
        print_err(''.join(s))

    if dry_run:
        print('dryrun ' + ''.join(s))
    else:
        os.system(sh_cmd)


subparser = subparsers.add_parser('exec')
subparser.add_argument('cmd-key', metavar='command-key')
subparser.add_argument('args', nargs='*', metavar='arguments')
subparser.add_argument('-d', '--dry-run', action='store_true')
subparser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction, default=bool(ENV_CTX_VERBOSE))
@handles(subparser)
def handle_exec(cmd_key, args, verbose, dry_run):
    import shlex
    import subprocess

    cmd = CTX[cmd_key][1]
    sh_cmd = shlex.split(cmd)
    sh_cmd.extend(args)

    s = ('exec command: ',
        STYLE['command'],
        repr(args),
        COLOR[''],
        )

    if verbose:
        print(''.join(s), file=sys.stderr)

    if dry_run:
        print('dryrun ' + ''.join(s))
    else:
        proc = subprocess.Popen(sh_cmd)
        return proc.wait()


subparser = subparsers.add_parser('keys')
@handles(subparser)
def handle_keys():
    keys = sorted(CTX.keys())
    for k in keys:
        s = (STYLE['key'],
            k,
            COLOR[''],
            )
        print(''.join(s))


subparser = subparsers.add_parser('items')
subparser.add_argument('keys', nargs='*', metavar='key')
@handles(subparser)
def handle_items(keys):
    # print out the items in creation order
    # if args, use args as keys

    # allow for `ctx items | ctx update -" to preserve time order
    if keys:
        items = [(CTX[k][0], k, CTX[k][1]) for k in keys]
    else:
        everything = [(v[0], k, v[1]) for k, v in CTX.items()]
        items = sorted(everything)

    # make the output resemble `env`
    for _, _key, _value in items:
        s = (STYLE['key'], _key, COLOR[''], '=',
             STYLE['value'],
             _value,
             COLOR['']
        )
        print(''.join(s))


subparser = subparsers.add_parser('log')
@handles(subparser)
def handle_log():
    for x in LOG:
        print(x)


subparser = subparsers.add_parser('import')
subparser.add_argument('env-key')  # TODO allow (and handle) multiple environment keys
subparser.add_argument('new-key', nargs='?')  # TODO refactor `new-key` argument into a formatting convention of each `env-key` argument (i.e. 'KEY[{=|:}NEWKEY])
@handles(subparser)
def handle_import(env_key, new_key):
    missing = object()
    env_value = os.environ.get(env_key, missing)
    store_as = new_key if new_key else env_key

    if env_value is not missing:
        CTX[store_as] = (NOW, env_value)


subparser = subparsers.add_parser('update')
subparser.add_argument('fid', type=argparse.FileType(), metavar='file')
@handles(subparser)
def handle_update(fid):
    # update the keys with the given file of key=value lines
    # example: $ env | ctx update -
    # the "items" command can be used to bulk transfer key-values
    #   ctx items > kv.txt
    #   ctx switch new_env
    #   ctx update kv.txt
    # process the lines
    d = {}
    now2 = NOW

    with fid:
        for line in fid.readlines():
            _key, eq, _value = line.partition('=')
            _value = _value.rstrip() # strip newline
            d[_key] = (now2, _value)
            LOG.append((now2, 'update_set', _key, _value))

            while True:  # ensure unique now
                _now2 = get_now()
                if _now2 != now2:
                    now2 = _now2
                    break

    # update if no error occurs
    CTX.update(d)


subparser = subparsers.add_parser('clear')
subparser.add_argument('name', choices=[CTX_NAME], metavar='name')  # name of current context required as a failsafe
@handles(subparser)
def handle_clear(name):
    assert name == CTX_NAME
    CTX.clear()


subparser = subparsers.add_parser('version')
@handles(subparser)
def handle_version():
    _print_version()


subparser = subparsers.add_parser('dict')
@handles(subparser)
def handle_dict():
    print(CTX_FILE)


subparser = subparsers.add_parser('help')
subparser.add_argument('cmd', nargs='?', choices=sorted(subparsers.keys()))
@handles(subparser)
def handle_help(cmd):
    if cmd:
        subparsers[cmd].print_help()
    else:
        parser.print_help()


def main(argv=sys.argv[1:]):
    args = parser.parse_args(argv)

    if ENV_CTX_VERBOSE == 1:
        _print_verbose()
    elif ENV_CTX_VERBOSE > 1:
        _print_debug(args)

    os.makedirs(CTX_HOME, exist_ok=True)
    retcode = handle(args) or 0

    LOG.save()
    CTX.save()
    parser.exit(retcode)


if __name__.endswith('__main__'):
    main()
