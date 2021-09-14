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

import inspect
import json
import os
import sys

from abc import ABC, abstractmethod
from argparse import Action, ArgumentParser, BooleanOptionalAction, Namespace
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from enum import Enum, auto
from functools import partial
from pathlib import Path
from types import FunctionType
from typing import Optional, Union

__version__ = '1.0.0-dev.0'

#region Configuration & Common Utilities

#region Environment Variables

ENV_CTX_COLOR: bool = bool(int(os.environ['CTX_COLOR'])) if 'CTX_COLOR' in os.environ else None
ENV_CTX_HOME: str = os.environ.get('CTX_HOME', None)
ENV_CTX_NAME: str = os.environ.get('CTX_NAME', None)
ENV_CTX_VERBOSE: int = int(os.environ.get('CTX_VERBOSE', 0))

#endregion

#region ANSI Coloring

CTX_COLOR = ENV_CTX_COLOR if ENV_CTX_COLOR is not None else False

class AnsiColor(Enum):
    BLACK  = '\033[0;30m'
    RED    = '\033[0;31m'
    GREEN  = '\033[0;32m'
    BLUE   = '\033[0;94m'
    YELLOW = '\033[0;33m'
    RESET  = '\033[0m'

    def __init__(self, ansi_code):
        self.ansi_code = ansi_code

    def format(self, text: str, *, reset=True):
        parts = [self.ansi_code, text]
        if reset:
            parts.append(self.RESET.ansi_code)
        return ''.join(parts)

class Style(Enum):
    KEY     = AnsiColor.GREEN
    VALUE   = AnsiColor.BLUE
    TIME    = AnsiColor.RED
    COMMAND = AnsiColor.BLUE
    CONTEXT = AnsiColor.BLUE
    VERSION = AnsiColor.RED
    WARNING = AnsiColor.RED
    RESET   = AnsiColor.RESET

    def __init__(self, color):
        self.color = color

    def format(self, text: str, *, reset=True):
        return self.color.format(text, reset=reset)

def format_key(key: str) -> str:
    return Style.KEY.format(key) if CTX_COLOR else key

def format_value(value: str) -> str:
    return Style.VALUE.format(value) if CTX_COLOR else value

def format_time(time: str) -> str:
    return Style.TIME.format(time) if CTX_COLOR else time

def format_command(command: str) -> str:
    return Style.COMMAND.format(command) if CTX_COLOR else command

def format_context(context: str) -> str:
    return Style.CONTEXT.format(context) if CTX_COLOR else context

def format_version(version: str) -> str:
    return Style.VERSION.format(version) if CTX_COLOR else version

def format_warning(warning: str) -> str:
    return Style.WARNING.format(warning) if CTX_COLOR else warning

#endregion

#region FileSystem Paths

CTX_HOME = Path(ENV_CTX_HOME) if ENV_CTX_HOME else Path.home() / '.ctx'
CTX_NAME_FILE = CTX_HOME / '_name.txt'

DEFAULT_CTX_NAME = 'main'

if ENV_CTX_NAME:
    CTX_NAME = ENV_CTX_NAME
elif CTX_NAME_FILE.exists():
    CTX_NAME = CTX_NAME_FILE.read_text().strip()
else:
    CTX_NAME = DEFAULT_CTX_NAME

CTX_FILE = CTX_HOME / f"{CTX_NAME}.json"
LOG_FILE = CTX_HOME / f"{CTX_NAME}.log"

#endregion

#region Data Access

class LazyData(ABC):
    def __init__(self, file: Path):
        self._file = file
        self.__data: Union[list, dict] = None
        self._modified = False

    @abstractmethod
    def _get_default_data(self):
        raise NotImplementedError()

    def _load(self):
        if self._file.exists():
            text = self._file.read_text(encoding='utf8')
            self.__data = json.loads(text)
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
        text = json.dumps(self._data, indent=4)
        self._file.write_text(text, encoding='utf8')

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

#endregion

#region Exceptions

class KeysError(KeyError):
    def __init__(self, keys: Iterable=None):
        super().__init__()
        if keys is None:
            self.keys = []
        else:
            self.keys = list(keys)

    def __str__(self):
        return str(self.keys)[1:-1]

class KeyExistsError(LookupError):
    pass

class KeysExistError(LookupError):
    def __init__(self, keys: Iterable=None):
        super().__init__()
        if keys is None:
            self.keys = []
        else:
            self.keys = list(keys)

    def __str__(self):
        return str(self.keys)[1:-1]

class ContextExistsError(KeyExistsError):
    pass

class DefaultContextError(ValueError):
    def __init__(self, name=DEFAULT_CTX_NAME):
        super().__init__(f"May not modify the default context: '{name}'")

#endregion

#region Printing Information

print_err = partial(print, file=sys.stderr)

def _print_version():
    print_err(''.join((
        'shellctx version ',
        format_version(__version__)
    )))

def _print_verbose():
    print_err('CTX_VERBOSE=%i' % ENV_CTX_VERBOSE)
    _print_version()
    if ENV_CTX_HOME:
        print_err('CTX_HOME=%s' % ENV_CTX_HOME)
    if ENV_CTX_NAME:
        print_err('CTX_NAME=%s' % ENV_CTX_NAME)

def _print_parsed_args(args: Namespace):
    print_err('parsed args:')
    args_D = vars(args)
    max_key_len = max(len(k) for k in args_D.keys())
    for (key, value) in args_D.items():
        if key.startswith('_'):
            continue
        print_err(''.join((
            '    ',
            format_key(key),
            ':',
            ' ' * (1 + (max_key_len - len(key))),
            format_value(repr(value)),
        )))

def _print_debug(args: Namespace):
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
    s = ['Using context ', format_context(CTX_NAME)]
    if ENV_CTX_NAME:
        s.append(' (set by CTX_NAME)')
    if ENV_CTX_HOME:
        s.append(' (from CTX_HOME={ENV_CTX_HOME})')
    print(''.join(s))
    print(''.join((
        'There are ',
        format_value(str(len(everything))),
        ' entries.'
    )))
    print()

    for ctime, _key, _value in x:
        print(''.join((
            format_time(ctime),
            '\t',
            format_key(_key),
            ' = ',
            format_value(str(_value)),
        )))

#endregion

#region Time

def now() -> str:
    return datetime.now().isoformat()

NOW = now()

#endregion

def first(it: Iterable):
    for x in it:
        return x

#endregion

#region Argument Parsing & Handling

#region Command Handling Infrastructure

def set_handler(parser: ArgumentParser, handler: Callable):
    parser.set_defaults(_handler=handler)

def handle(args: Namespace) -> Optional[int]:
    args._handler(args)

def handles(parser: ArgumentParser) -> Callable[[FunctionType], None]:
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
    return set_wrapped_handler

#endregion

class AbbreviatableBooleanOptionalAction(Action):
    # assumes that the only prefix char is '-'
    # assumes that any long option is prefixed with exactly '--'
    # assumes that any short option is prefixed with exactly '-'

    NARGS = 0

    @classmethod
    def _validate_option_strings(cls, option_strings: Sequence[str]) -> None:
        assert not len(option_strings) == 2 or option_strings[0] > option_strings[1], "option strings are not sorted"
        valid = (
            len(option_strings) == 1
            and option_strings[0].startswith('--')
        ) or (
            len(option_strings) == 2
            and not option_strings[0].startswith('--')  # no more than 1 long option string
            and option_strings[1].startswith('--')  # no more than 1 short option string
        )
        if not valid:
            raise ValueError(f'{cls.__name__} must be defined with no more than 1 short option string and exactly 1 long option string')

    @classmethod
    def _extended_option_strings(cls, option_strings: Sequence[str]) -> list[str]:
        cls._validate_option_strings(option_strings)

        # ensures any short option is before the long option
        # also copies the sequence for encapsulation purposes
        option_strings: list[str] = sorted(option_strings)

        long_option = option_strings[-1]
        negated_long_option = '--no-' + long_option[2:]
        option_strings.append(negated_long_option)

        if option_strings[0].startswith('--'):  # there is not an explicit short option
            short_option = '-' + long_option[2]
            option_strings.insert(0, short_option)
        else:
            short_option = option_strings[0]
        negated_short_option = short_option.swapcase()
        option_strings.insert(1, negated_short_option)

        return option_strings

    def __init__(self, option_strings: Sequence[str], dest: str, **kwargs):
        option_strings = self._extended_option_strings(option_strings)
        assert len(option_strings) == 4, f"{option_strings=}"

        self.short_option_strings = option_strings[:2]
        self.long_option_strings = option_strings[2:]
        self.positive_option_strings = option_strings[0::2]
        self.negative_option_strings = option_strings[1::2]

        kwargs['nargs'] = self.NARGS
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser: ArgumentParser, namespace: Namespace, values: Sequence, option_string: str=None) -> None:
        if option_string in self.positive_option_strings:
            value = True
        elif option_string in self.negative_option_strings:
            value = False
        else:
            raise NotImplementedError(f"{option_string=}")
        setattr(namespace, self.dest, value)

    def format_usage(self):
        return ' | '.join(self.short_option_strings)

class ColorAction(AbbreviatableBooleanOptionalAction):
    def __call__(self, parser: ArgumentParser, namespace: Namespace, values: Sequence, option_string: str=None) -> None:
        super().__call__(parser, namespace, values, option_string)
        global CTX_COLOR
        CTX_COLOR = getattr(namespace, self.dest)

global_options = ArgumentParser(add_help=False, allow_abbrev=False)
group = global_options.add_argument_group('global options')
group.add_argument('--color', action=ColorAction, default=CTX_COLOR)
group.add_argument('--verbose', action=BooleanOptionalAction, default=ENV_CTX_VERBOSE)


parser = ArgumentParser(prog='ctx', allow_abbrev=False, parents=[global_options])
parser.add_argument('-v', '--version', action='store_true')
@handles(parser)
def handle_(version: bool, verbose: bool):
    if version:
        if verbose:
            _print_version()
        else:
            print(__version__)
    else:
        _print_full_items()

# TODO pass `help` (brief) for each parser/subparser
# TODO pass `description` (full) for each parser/subparser
# TODO pass `help` for each parser/subparser argument

subparsers = parser.add_subparsers(title='commands')
type(subparsers).__getitem__ = lambda self, key: self._name_parser_map[key]
type(subparsers).keys = lambda self: self._name_parser_map.keys()


class GetType(Enum):
    DICT = auto()
    ITEMS = auto()
    KEYS = auto()
    VALUES = auto()

class MissingAction(Enum):
    FORCE = auto()
    SKIP = auto()

subparser = subparsers.add_parser('get', aliases=['g'], parents=[global_options])
subparser.add_argument('keys', nargs='*', metavar='key')
group = subparser.add_argument_group('missing key handling').add_mutually_exclusive_group()
group.add_argument('-f', '--force', action='store_const', const=MissingAction.FORCE, dest='missing_action')
group.add_argument('-s', '--skip-missing', action='store_const', const=MissingAction.SKIP, dest='missing_action')
group = subparser.add_argument_group('return value formatting').add_mutually_exclusive_group()
group.add_argument('-d', '--dict', action='store_const', const=GetType.DICT, dest='get_type')
group.add_argument('-i', '--item', '--items', action='store_const', const=GetType.ITEMS, dest='get_type')
group.add_argument('-k', '--key', '--keys', action='store_const', const=GetType.KEYS, dest='get_type')
group.add_argument('-v', '--value', '--values', action='store_const', const=GetType.VALUES, dest='get_type')
@handles(subparser)
def handle_get(keys: list[str], missing_action: Optional[MissingAction], get_type: Optional[GetType]):
    if not keys:
        if not get_type:
            print_err(format_warning(
                'Error: Must supply at least one key and/or one of the following options: '
                '--d(ict(s)), --i(tem(s)), --k(ey(s)), xor --v(alue(s)).'
            ))
            return 1

        entries = sorted((i[1][0], i[0], i[1][1]) for i in CTX.items())  # (timestamp, key, value)
        items = (e[1:] for e in entries)

        if get_type == GetType.DICT:
            print(json.dumps(dict(items)))
        elif get_type == GetType.ITEMS:
            for key, value in items:
                print(''.join((
                    format_key(key),
                    '=',
                    format_value(value)
                )))
        elif get_type == GetType.KEYS:
            for key, _ in items:
                print(format_key(key))
        elif get_type == GetType.VALUES:
            for _, value in items:
                print(format_value(value))
        else:
            raise NotImplementedError(get_type)
        return

    if missing_action == MissingAction.FORCE:
        items = ((k, CTX[k][1] if k in CTX else '') for k in keys)
    elif missing_action == MissingAction.SKIP:
        items = ((k, CTX[k][1]) for k in keys if k in CTX)
    elif not missing_action:
        missing_keys = {k for k in keys if k not in CTX}
        if len(missing_keys) == 1:
            raise KeyError(first(missing_keys))
        elif len(missing_keys) > 1:
            raise KeysError(sorted(missing_keys))
        items = ((k, CTX[k][1]) for k in keys)
    else:
        raise NotImplementedError(missing_action)

    if get_type == GetType.DICT:
        print(json.dumps(dict(items)))
    elif get_type == GetType.ITEMS:
        for key, value in items:
            print(''.join((
                format_key(key),
                '=',
                format_value(value)
            )))
    elif get_type == GetType.KEYS:
        for key, _ in items:
            print(format_key(key))
    elif get_type == GetType.VALUES or not get_type:
        for _, value in items:
            print(format_value(value))
    else:
        raise NotImplementedError(get_type)


subparser = subparsers.add_parser('set', aliases=['s'], parents=[global_options])
subparser.add_argument('keys', nargs='+', metavar='key')
subparser.add_argument('value')
group = subparser.add_argument_group('key options').add_mutually_exclusive_group()
group.add_argument('-e', '--entry', '--entries', action='store_true')
group.add_argument('-n', '--no-clobber', action='store_true')
group = subparser.add_argument_group('value options')
group.add_argument('-p', '--path', action='store_true')
@handles(subparser)
def handle_set(keys: list[str], value: str, path: bool, entry: bool, no_clobber: bool, verbose: bool):
    if path:
        base = Path.cwd()
        value = base if value == '.' else base / value

    if entry:
        new_keys = []
        for key in keys:
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

            new_key = prefix + ('%03i' % next_num)
            assert(new_key not in CTX)

            new_keys.append(new_key)

    for key in keys:
        if no_clobber and key in CTX:
            if verbose:
                print_err(f"KeyExistsError: {key}")
            continue

        LOG.append(NOW, 'set', key, value)
        CTX[key] = (NOW, value)

        if verbose:
            print(''.join((
                format_key(key),
                '=',
                format_value(value),
            )))


subparser = subparsers.add_parser('del', aliases=['d'], parents=[global_options])
subparser.add_argument('keys', nargs='+', metavar='key')
@handles(subparser)
def handle_del(keys: list[str], verbose: bool):
    removed_keys = set()
    for key in keys:
        if key in removed_keys:
            if verbose:
                print_err(f"KeyError: {key}")
            continue

        LOG.append(NOW, 'del', key)
        del CTX[key]

        if verbose:
            print(''.join((
                format_key(key),
                '=',
            )))

        removed_keys.add(key)


def _get_contexts(include_default=True):
    files = list(CTX_HOME.iterdir())
    ext = '.json'
    names = [f.stem for f in files if f.suffix == ext]
    if CTX_NAME not in names:
        names.append(CTX_NAME)
    if not include_default:
        try:
            names.remove(DEFAULT_CTX_NAME)
        except:
            pass
    names.sort()
    return names

class GetContextType(Enum):
    ALL = auto()
    CURRENT = auto()
    DEFAULT = auto()
    USER = auto()

subparser = subparsers.add_parser('get-ctx', aliases=['gc'], parents=[global_options])
group = subparser.add_argument_group('context selection').add_mutually_exclusive_group()
group.set_defaults(get_context_type=GetContextType.CURRENT)
group.add_argument('-a', '--all', action='store_const', const=GetContextType.ALL, dest='get_context_type')
group.add_argument('-d', '--default', action='store_const', const=GetContextType.DEFAULT, dest='get_context_type')
group.add_argument('-u', '--user', action='store_const', const=GetContextType.USER, dest='get_context_type')
@handles(subparser)
def handle_get_ctx(get_context_type: GetContextType, verbose: bool):
    if get_context_type == GetContextType.ALL:
        contexts = _get_contexts()
        if verbose:
            for name in contexts:
                print(''.join((
                    '* ' if name == CTX_NAME else '  ',
                    format_context(name),
                )))
        else:
            for name in contexts:
                print(format_context(name))
    elif get_context_type == GetContextType.CURRENT:
        print(format_context(CTX_NAME))
    elif get_context_type == GetContextType.DEFAULT:
        print(format_context(DEFAULT_CTX_NAME))
    elif get_context_type == GetContextType.USER:
        contexts = _get_contexts(include_default=False)
        if verbose:
            for name in contexts:
                print(''.join((
                    '* ' if name == CTX_NAME else '  ',
                    format_context(name)
                )))
        else:
            for name in contexts:
                print(format_context(name))
    else:
        raise NotImplementedError(get_context_type)



def _set_ctx(name):
    with open(CTX_NAME_FILE, 'w') as fid:
        fid.write(name)

subparser = subparsers.add_parser('set-ctx', aliases=['sc'], parents=[global_options])
subparser.add_argument('name', nargs='?', default=DEFAULT_CTX_NAME)
@handles(subparser)
def handle_set_ctx(name: Optional[str], verbose: bool):
    if ENV_CTX_NAME and name != ENV_CTX_NAME:
        print_err(''.join((
            format_warning('context set by CTX_NAME as '),
            format_context(ENV_CTX_NAME),
            format_warning('. Not switching.'),
        )))
        return 1

    if name != CTX_NAME:
        _set_ctx(name)

    if verbose:
        print(''.join(('switching to "',
            format_context(name),
            '" from "',
            format_context(CTX_NAME),
            '"',
        )))


subparser = subparsers.add_parser('del-ctx', aliases=['dc'], parents=[global_options])
subparser.add_argument('names', nargs='+', metavar='name')
@handles(subparser)
def handle_del_ctx(names: list[str]):
    assert len(names) > 0

    retcode = 0
    for name in names:
        if name == DEFAULT_CTX_NAME:
            err = DefaultContextError()
            if len(names) == 1:
                raise err
            else:
                assert len(names) > 1
                print_err(str(err))
                retcode = 1
                continue

        ctx_file = CTX_HOME / f"{name}.json"
        ctx_file.unlink(missing_ok=True)

        log_file = CTX_HOME / f"{name}.log"
        log_file.unlink(missing_ok=True)

        if name == CTX_NAME:
            _set_ctx(DEFAULT_CTX_NAME)
    return retcode


subparser = subparsers.add_parser('version', parents=[global_options])
@handles(subparser)
def handle_version():
    _print_version()


# The 'help' subparser must be the last one defined because of the need to
# specify the names of all others as the choices.
subparser = subparsers.add_parser('help', parents=[global_options])
subparser.add_argument('cmd', nargs='?', choices=sorted(subparsers.keys()))
@handles(subparser)
def handle_help(cmd: Optional[str]):
    if cmd:
        subparsers[cmd].print_help()
    else:
        parser.print_help()

#endregion

#region Main

def main(args=sys.argv[1:]):
    _args = parser.parse_args(args)

    if ENV_CTX_VERBOSE == 1:
        _print_verbose()
    elif ENV_CTX_VERBOSE > 1:
        _print_debug(_args)

    CTX_HOME.mkdir(parents=True, exist_ok=True)
    retcode = handle(_args) or 0

    LOG.save()
    CTX.save()
    parser.exit(retcode)

if __name__.endswith('__main__'):
    main()

#endregion
