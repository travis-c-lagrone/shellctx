#!/usr/bin/env python3
"""
tclg-shellctx
--------

Shell Context Helper

Repository: https://github.com/travis-c-lagrone/python-tclg-shellctx
Author:     Travis C. LaGrone
Date:       2021-09-15
License:    GNU GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html

Original Repository: https://github.com/serwy/shellctx
Original Author:     Roger D. Serwy
Original License:    GNU GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html

"""

import inspect
import json
import os
import sys

from abc import ABC, abstractmethod
from argparse import (Action, ArgumentParser, BooleanOptionalAction,
                      Namespace, RawDescriptionHelpFormatter,)
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from enum import Enum, auto
from functools import partial, wraps
from pathlib import Path
from types import FunctionType
from typing import Optional, TypeVar, Union


__version__ = '1.0.0-alpha.1'

#region Configuration & Common Utilities

#region Environment Variables

T = TypeVar('T')
def get_environ_variable(name: str, type: Callable[[str], T]=str) -> Optional[T]:
    value = None
    if name in os.environ and os.environ[name]:
        value = type(os.environ[name])
    return value

def bool_int(x: str) -> bool:
    return bool(int(x))

ENV_CTX_COLOR: Optional[bool] = get_environ_variable('CTX_COLOR', type=bool_int)
ENV_CTX_DEBUG: Optional[bool] = get_environ_variable('CTX_DEBUG', type=bool_int)
ENV_CTX_HOME: Optional[str] = get_environ_variable('CTX_HOME')
ENV_CTX_NAME: Optional[str] = get_environ_variable('CTX_NAME')
ENV_CTX_VERBOSE: Optional[bool] = get_environ_variable('CTX_VERBOSE', type=bool_int)

ENV_DOCS = {
    'CTX_COLOR': 'set the --color/--no-color global option (int: 1 or 0)',
    'CTX_DEBUG': 'set the --debug/--no-debug global option (int: 1 or 0)',
    'CTX_HOME': 'the directory in which this program persists data (str: absolute path)',
    'CTX_NAME': 'the name of the context to use as the current context (str)',
    'CTX_VERBOSE': 'set the --verbose/--no-verbose global option (int: 1 or 0)',
}

#endregion

#region ANSI Coloring

CTX_COLOR: bool = ENV_CTX_COLOR if ENV_CTX_COLOR is not None else False

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

CTX_VERBOSE: bool = ENV_CTX_VERBOSE if ENV_CTX_VERBOSE is not None else False
CTX_DEBUG: bool = ENV_CTX_DEBUG if ENV_CTX_DEBUG is not None else False

print_err = partial(print, file=sys.stderr)

def print_debug(args: Namespace):
    print_err(''.join((
        format_key('CTX_VERSION'),
        ' = ',
        format_version(repr(__version__))
    )))
    print_err()

    for prefix in ('ENV_', ''):
        for key in ENV_DOCS.keys():
            name = prefix + key
            value = globals()[name]
            print_err(''.join((
                format_key(name),
                ' = ',
                format_value(repr(value))
            )))
        print_err()

    for name in dir(args):
        if (
            name.startswith('__')
            or (
                name.startswith('_')
                and (
                    inspect.ismethod(attr := getattr(args, name))
                    or inspect.ismethoddescriptor(attr)))
        ):
            continue

        value = getattr(args, name)
        print_err(''.join((
            format_key(f'args.{name}'),
            ' = ',
            format_value(repr(value)),
        )))
    print_err()

def print_full_items():
    # timestamp, key, value
    everything = [(v[0], k, v[1]) for k, v in CTX.items()]
    x = sorted(everything, reverse=True)
    s = ['Using context ', format_context(CTX_NAME)]
    if ENV_CTX_NAME:
        s.append(' (set by CTX_NAME)')
    if ENV_CTX_HOME:
        s.append(f" (from CTX_HOME={ENV_CTX_HOME})")
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
        @wraps(handler)
        def wrapped(args):
            kwargs = {k: getattr(args, k) for k in keys}
            if has__args__:
                kwargs['__args__'] = args
            return handler(**kwargs)
        set_handler(parser, wrapped)
        return wrapped
    return set_wrapped_handler

#endregion

#region Global Option Infrastructure

def global_option_action(cls: Action):
    __call__ = cls.__call__  # the original unwrapped and unbound __call__ implementation
    @wraps(__call__)
    def wrapped(self, parser: ArgumentParser, namespace: Namespace, values: Sequence, option_string: str=None) -> None:
        if not hasattr(namespace, '_defaulted_globals'):
            namespace._explicitly_set_globals = set()
        if self.dest not in namespace._explicitly_set_globals:
            __call__(self, parser, namespace, values, option_string)
            namespace._explicitly_set_globals.add(self.dest)
    cls.__call__ = wrapped
    return cls

@global_option_action
class GlobalBooleanOptionalAction(BooleanOptionalAction):
    pass

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

@global_option_action
class GlobalAbbreviatableBooleanOptionalAction(AbbreviatableBooleanOptionalAction):
    pass

#endregion

class GlobalColorAction(GlobalAbbreviatableBooleanOptionalAction):
    def __call__(self, parser: ArgumentParser, namespace: Namespace, values: Sequence, option_string: str=None) -> None:
        super().__call__(parser, namespace, values, option_string)
        global CTX_COLOR
        CTX_COLOR = getattr(namespace, self.dest)

class GlobalVerboseAction(GlobalBooleanOptionalAction):
    def __call__(self, parser: ArgumentParser, namespace: Namespace, values: Sequence, option_string: str=None) -> None:
        super().__call__(parser, namespace, values, option_string)
        global CTX_VERBOSE
        CTX_VERBOSE = getattr(namespace, self.dest)

class GlobalDebugAction(GlobalBooleanOptionalAction):
    def __call__(self, parser: ArgumentParser, namespace: Namespace, values: Sequence, option_string: str=None) -> None:
        super().__call__(parser, namespace, values, option_string)
        global CTX_DEBUG
        CTX_DEBUG = getattr(namespace, self.dest)

global_options = ArgumentParser(add_help=False, allow_abbrev=False)
global_options.add_argument('--color',
        action=GlobalColorAction, default=CTX_COLOR,
        help=f"(default: {CTX_COLOR}{', as set by CTX_COLOR' if CTX_COLOR else ''})")
global_options.add_argument('--verbose',
        action=GlobalVerboseAction, default=CTX_VERBOSE)
global_options.add_argument('--debug',
        action=GlobalDebugAction, default=CTX_DEBUG)


parser = ArgumentParser(prog='ctx',
        allow_abbrev=False, parents=[global_options],
        # usage='%(prog)s [COMMAND]',
        description=' '.join((
            'Manage key-value pairs organized by implicit user-defined contexts.',
            'Without any command, shows the current context and its entries.')))
parser.add_argument('-v', '--version',
        action='store_true',
        help='show the version of this program and exit')
@handles(parser)
def handle_(version: bool, verbose: bool):
    if version:
        print(format_version(__version__))
    else:
        print_full_items()


subparsers = parser.add_subparsers(metavar='COMMAND')
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

subparser = subparsers.add_parser('get',
        aliases=['g'], parents=[global_options],
        help='get the value for the key(s)',
        usage='ctx get [KEY ...] [-f | -s] [-d | -i | -k | -v]',
        description=' '.join((
            'Get a representation of each key (if any).',
            'If no format option is specified, then the value of each key is returned.',
            'If no missing key option is specified, then an error is returned if any key is missing.',
            'If no keys are specified, then a format option must be specified and all keys are returned.')))
subparser.add_argument('keys',
        nargs='*', metavar='KEY',
        help='the key(s) whose value to return')
group = subparser.add_argument_group('missing key handling').add_mutually_exclusive_group()
group.add_argument('-f', '--force',
        action='store_const', const=MissingAction.FORCE, dest='missing_action',
        help='return an empty string for any missing key')
group.add_argument('-s', '--skip-missing',
        action='store_const', const=MissingAction.SKIP, dest='missing_action',
        help='skip outputting any missing key')
group = subparser.add_argument_group('return value formatting').add_mutually_exclusive_group()
group.add_argument('-d', '--dict',
        action='store_const', const=GetType.DICT, dest='get_type',
        help='return all key-value pairs as a dictionary (JSON object)')
group.add_argument('-i', '--items',
        action='store_const', const=GetType.ITEMS, dest='get_type',
        help='return all key-value pairs as items (i.e. line-oriented string data) (e.g. \'foo=bar\')')
group.add_argument('-k', '--keys',
        action='store_const', const=GetType.KEYS, dest='get_type',
        help='return all keys (no values)')
group.add_argument('-v', '--values',
        action='store_const', const=GetType.VALUES, dest='get_type',
        help='return all values (no keys)')
@handles(subparser)
def handle_get(keys: list[str], missing_action: Optional[MissingAction], get_type: Optional[GetType]):
    if not keys:
        if not get_type:
            print_err(format_warning(
                'Error: Must supply at least one key and/or one of the following options: '
                '-d/--dict, -i/--items, -k/--keys, xor -v/--values.'
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


subparser = subparsers.add_parser('set',
        aliases=['s'], parents=[global_options],
        help='set the value for the key(s)',
        usage='ctx set KEY [KEY ...] VALUE [-e | -n] [-p]',
        description=' '.join((
            'Set each key to the value.',
            'Neither any key nor the value may be an empty string.',
            'Any pre-existing key is overwritten without warning, unless the --no-clobber option is specified.')))
subparser.add_argument('keys',
        nargs='+', metavar='KEY',
        help='the key(s) whose value to set')
subparser.add_argument('value',
        metavar='VALUE',
        help='the value to set for the key(s); may not be the empty string')
group = subparser.add_argument_group('key options').add_mutually_exclusive_group()
group.add_argument('-e', '--entries',
        action='store_true', dest='entry',
        help='append a monotonically increasing non-negative integer to each key as if it were an indexed entry in an array (e.g. \'actions\' -> \'action_000\'')
group.add_argument('-n', '--no-clobber',
        action='store_true',
        help='skip setting any key that is already defined')
group = subparser.add_argument_group('value options')
group.add_argument('-p', '--path',
        action='store_true',
        help='prepend the current working directory to the value')
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
            suffixes = [k[N:] for k in CTX.keys() if k.startswith(prefix)]

            max_num = 0 if key in CTX else -1
            for num in suffixes:
                try:
                    num = int(num)
                except:
                    pass
                else:
                    max_num = max(num, max_num)
            next_num = max_num + 1

            new_key = prefix + str(next_num)
            assert(new_key not in CTX)

            new_keys.append(new_key)
        keys = new_keys

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


subparser = subparsers.add_parser('del',
        aliases=['d'], parents=[global_options],
        help='delete the key(s)',
        usage='ctx del KEY [KEY ...]',
        description=' '.join((
            'Delete each key and its value.',
            'Missing keys are skipped without error.')))
subparser.add_argument('keys',
        nargs='+', metavar='KEY',
        help='the key(s) to delete (if they exist)')
@handles(subparser)
def handle_del(keys: list[str], verbose: bool):
    removed_keys = set()
    for key in keys:
        if key in removed_keys:
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

subparser = subparsers.add_parser('get-ctx',
        aliases=['gc'], parents=[global_options],
        usage='ctx get-ctx [-a | -r | -d | -u]',
        help='get the context name(s)',
        description=' '.join((
            'Get the context name(s).',
            'If no selection option is specified, the name of the current (active) context is returned.')))
group = subparser.add_argument_group('context selection').add_mutually_exclusive_group()
group.set_defaults(get_context_type=GetContextType.CURRENT)
group.add_argument('-a', '--all',
        action='store_const', const=GetContextType.ALL, dest='get_context_type',
        help='get all context names')
group.add_argument('-r', '--current',
        action='store_const', const=GetContextType.CURRENT, dest='get_context_type',
        help='get the current (active) context name')
group.add_argument('-d', '--default',
        action='store_const', const=GetContextType.DEFAULT, dest='get_context_type',
        help='get the (built-in) default context name')
group.add_argument('-u', '--user',
        action='store_const', const=GetContextType.USER, dest='get_context_type',
        help='get all user-defined context names, if any (excludes the default context name)')
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

subparser = subparsers.add_parser('set-ctx',
        aliases=['sc'], parents=[global_options],
        help='switch between contexts',
        usage='ctx set-ctx [NAME]',
        description=' '.join((
            'Switch the current (active) context to the named context.',
            f'Without any name, switches to the default context: {DEFAULT_CTX_NAME}.',
            'It is not an error to switch to the same context that is already active.',
            'If the named context does not exist, it is created prior to switching to it.')))
subparser.add_argument('name',
        nargs='?', default=DEFAULT_CTX_NAME, metavar='NAME',
        help='the name of the context to switch to; it will be created if it does not already exist')
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
        print(''.join(('switching to ',
            format_context(name),
            ' from ',
            format_context(CTX_NAME),
            '',
        )))


subparser = subparsers.add_parser('del-ctx',
        aliases=['dc'], parents=[global_options],
        help='delete the named context',
        usage='ctx del-ctx NAME [NAME ...]',
        description=' '.join((
            'Delete the named context(s).',
            'A single dot (.) is considered an alias for the current context.',
            f'The default context ({DEFAULT_CTX_NAME}) may not be deleted.',
            'It is not an error to delete a non-existent context.')))
subparser.add_argument('names',
        nargs='+', metavar='NAME',
        help='; '.join((
            'the context(s) to delete',
            'a dot (.) denotes the current context',
            f'may not be the default context ({DEFAULT_CTX_NAME})')))
@handles(subparser)
def handle_del_ctx(names: list[str], verbose: bool):
    assert len(names) > 0

    names = [CTX_NAME if n == '.' else n for n in names]

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
        log_file = CTX_HOME / f"{name}.log"

        ctx_file.unlink(missing_ok=True)
        log_file.unlink(missing_ok=True)
        if verbose:
            print_err('Deleted context ' + format_context(name))

        if name == CTX_NAME:
            _set_ctx(DEFAULT_CTX_NAME)
            if verbose:
                print_err(''.join((
                    'Switched to the default context (',
                    format_context(DEFAULT_CTX_NAME),
                    ') because the current context was deleted'
                )))
    return retcode


subparser = subparsers.add_parser('env',
        parents=[global_options], formatter_class=RawDescriptionHelpFormatter,
        help='show all relevant environment variables',
        usage='ctx env',
        description='\n\n'.join((
            ' '.join((
                'Show all environment variables (and their values) that configure this program.',
                'Any environment variable that is not defined is shown with the empty string as its value.')),
            '\n'.join((
                'environment variables:',
                *[var.ljust(22).rjust(24) + doc for var, doc in ENV_DOCS.items()]
            )))))
@handles(subparser)
def handle_env(verbose: bool):
    for name in ENV_DOCS.keys():
        value = globals()[f"ENV_{name}"]
        print(''.join((
            format_key(name),
            '=',
            format_value(str(value)) if value else ''
        )))


subparser = subparsers.add_parser('version',
        aliases=['v'], parents=[global_options],
        help='show the version of this program and exit',
        usage='ctx version',
        description='Return the current installed version of this program.')
@handles(subparser)
def handle_version():
    print(format_version(__version__))


# The 'help' subparser must be the last one defined because of the need to
# specify the names of all others as the choices.
subparser = subparsers.add_parser('help',
        aliases=['h'], parents=[global_options],
        help='show the help message for a command and exit',
        usage='ctx help [COMMAND]',
        description=' '.join((
            'Show the help message for the specified command (or command alias).',
            'If no command is specified, then show the help message for this program.')))
subparser.add_argument('cmd',
        nargs='?', choices=list(subparsers.keys()), metavar='COMMAND',
        help='the command (or alias) for which to show its help message')
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

    if CTX_DEBUG:
        print_debug(_args)

    CTX_HOME.mkdir(parents=True, exist_ok=True)
    retcode = handle(_args) or 0

    LOG.save()
    CTX.save()
    parser.exit(retcode)

if __name__.endswith('__main__'):
    main()

#endregion
