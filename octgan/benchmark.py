"""Main octgan benchmarking module."""

import os
import io
import types
import logging
import multiprocessing as mp

import tqdm
import psutil
import humanfriendly

from octgan.data import load_dataset
from octgan.evaluate import compute_scores
from octgan.results import make_leaderboard
from octgan.synthesizers.base import BaseSynthesizer


LOGGER = logging.getLogger(__name__)


DEFAULT_DATASETS = [
    "adult"
]


class TqdmLogger(io.StringIO):

    _buffer = ''

    def write(self, buf):
        self._buffer = buf.strip('\r\n\t ')

    def flush(self):
        LOGGER.info(self._buffer)


def _used_memory():
    process = psutil.Process(os.getpid())
    return humanfriendly.format_size(process.memory_info().rss)


def _score_synthesizer_on_dataset(name, synthesizer, iteration, args):
    try:
        LOGGER.info('Evaluating %s on dataset %s; iteration %s; %s',
                    name, args.dataset_name, iteration, _used_memory())

        train, test, meta, categoricals, ordinals = load_dataset(args.dataset_name, benchmark=True)
        if isinstance(synthesizer, type) and issubclass(synthesizer, BaseSynthesizer):
            synthesizer = synthesizer(dataset_name=args.dataset_name, args=args).fit_sample

        synthesized = synthesizer(train, categoricals, ordinals) 
        scores = compute_scores(test, synthesized, meta)
         
        return scores 

    except Exception:
        LOGGER.exception('Error running %s on dataset %s; iteration %s',
                         name, args.dataset_name, iteration)

    finally:
        LOGGER.info('Finished %s on dataset %s; iteration %s; %s',
                    name, args.dataset_name, iteration, _used_memory())


def _score_synthesizer_on_dataset_args(args):
    return _score_synthesizer_on_dataset(*args)


def _get_synthesizer_name(synthesizer):
    """Get the name of the synthesizer function or class.

    If the given synthesizer is a function, return its name.
    If it is a method, return the name of the class to which
    the method belongs.

    Args:
        synthesizer (function or method):
            The synthesizer function or method.

    Returns:
        str:
            Name of the function or the class to which the method belongs.
    """
    if isinstance(synthesizer, types.MethodType):
        synthesizer_name = synthesizer.__self__.__class__.__name__
    else:
        synthesizer_name = synthesizer.__name__

    return synthesizer_name


def _get_synthesizers(synthesizers):
    """Get the dict of synthesizers from the input value.

    If the input is a synthesizer or an iterable of synthesizers, get their names
    and put them on a dict.

    Args:
        synthesizers (function, class, list, tuple or dict):
            A synthesizer (function or method or class) or an iterable of synthesizers
            or a dict containing synthesizer names as keys and synthesizers as values.

    Returns:
        dict[str, function]:
            dict containing synthesizer names as keys and function as values.

    Raises:
        TypeError:
            if neither a synthesizer or an iterable or a dict is passed.
    """
    if callable(synthesizers):
        synthesizers = {_get_synthesizer_name(synthesizers): synthesizers}
    if isinstance(synthesizers, (list, tuple)):
        synthesizers = {
            _get_synthesizer_name(synthesizer): synthesizer
            for synthesizer in synthesizers
        }
    elif not isinstance(synthesizers, dict):
        raise TypeError('`synthesizers` can only be a function, a class, a list or a dict')

    return synthesizers


def run(synthesizers=None, iterations=1, workers=1, output_path=None, arguments=None):
    """Run the octgan benchmark and return a leaderboard."""

    scorer_args = list()

    if synthesizers != None:
        synthesizers = _get_synthesizers(synthesizers)

        for synthesizer_name, synthesizer in synthesizers.items():
            for iteration in range(iterations):
                args = (synthesizer_name, synthesizer, iteration, arguments)
                scorer_args.append(args)

    else:
        for iteration in range(iterations):
            args = ("benchmark", None, iteration, arguments)
            scorer_args.append(args)

    if workers in (0, 1):
        scores = map(_score_synthesizer_on_dataset_args, scorer_args)
    else:
        pool = mp.Pool(mp.cpu_count())
        scores = pool.imap_unordered(_score_synthesizer_on_dataset_args, scorer_args)

    scores = tqdm.tqdm(scores, total=len(scorer_args), file=TqdmLogger())


    return make_leaderboard(
        scores,
        output_path=output_path,
        args=arguments
    )