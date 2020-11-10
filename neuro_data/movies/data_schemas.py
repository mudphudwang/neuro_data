import io
import os
from collections import OrderedDict
from itertools import count
from pprint import pformat

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from neuro_data.movies.transforms import Subsequence
from .mixins import TraceMixin, BehaviorMixin
from .schema_bridge import *
from .. import logger as log
from ..utils.data import SplineMovie, FilterMixin, SplineCurve, NaNSpline, \
    fill_nans, h5cached, key_hash, save_dict_to_hdf5


dj.config['stores'] = {
    'scratch09': dict(
        protocol='file',
        location='/mnt/scratch09/djexternal/')
}


STACKS = [
    dict(animal_id=17977, stack_session=2, stack_idx=8, pipe_version=1, volume_id=1, registration_method=2),
    dict(animal_id=17886, stack_session=2, stack_idx=1, pipe_version=1, volume_id=1, registration_method=2),
    dict(animal_id=17797, stack_session=6, stack_idx=9, pipe_version=1, volume_id=1, registration_method=2),
    dict(animal_id=17795, stack_session=3, stack_idx=1, pipe_version=1, volume_id=1, registration_method=2),
]

UNIQUE_CLIP = {
    'stimulus.Clip': ('movie_name', 'clip_number', 'cut_after', 'skip_time'),
    'stimulus.Monet': ('rng_seed',),
    'stimulus.Monet2': ('rng_seed',),
    'stimulus.Trippy': ('rng_seed',),
    'stimulus.Matisse2': ('condition_hash',)
}

schema = dj.schema('neuro_data_movies', locals())

MOVIESCANS = [  # '(animal_id=16278 and session=11 and scan_idx between 5 and 9)',  # hollymonet
    # '(animal_id=15685 and session=2 and scan_idx between 11 and 15)',  # hollymonet
    'animal_id=17021 and session=18 and scan_idx=11',  # platinum (scan_idx in (10, 14))
    'animal_id=9771 and session=1 and scan_idx in (1,2)',  # madmonet
    'animal_id=17871 and session=4 and scan_idx=13',  # palindrome mouse
    'animal_id=17358 and session=5 and scan_idx=3',  # platinum
    'animal_id=17358 and session=9 and scan_idx=1',  # platinum
    platinum.CuratedScan() & dict(animal_id=18142, scan_purpose='trainable_platinum_classic', score=4),
    platinum.CuratedScan() & dict(animal_id=17797, scan_purpose='trainable_platinum_classic') & 'score > 2',
    'animal_id=16314 and session=3 and scan_idx=1',
    # golden
    experiment.Scan() & (stimulus.Trial & stimulus.Condition() & stimulus.Monet()) & dict(animal_id=8973),
    'animal_id=18979 and session=2 and scan_idx=7',
    'animal_id=18799 and session=3 and scan_idx=14',
    'animal_id=18799 and session=4 and scan_idx=18',
    'animal_id=18979 and session=2 and scan_idx=5',
    # start with segmentation method 6
    'animal_id=20457 and session=1 and scan_idx=15',
    'animal_id=20457 and session=2 and scan_idx=20',
    'animal_id=20501 and session=1 and scan_idx=10',
    'animal_id=20458 and session=3 and scan_idx=5',
    # manolis data
    'animal_id=16314 and session=4 and scan_idx=3',
    'animal_id=21067 and session=8 and scan_idx=9',
]


@schema
class MovieScanCandidate(dj.Manual):
    definition = """
    -> fuse.ScanDone
    ---
    candidate_note='': varchar(1024)  # notes about the scan (if any)
    """


@schema
class MovieScan(dj.Computed):
    definition = """
    # smaller primary key table for data

    -> MovieScanCandidate
    ---
    """

    class Unit(dj.Part):
        definition = """
        # smaller primary key table for data
        -> master
        unit_id              : int                          # unique per scan & segmentation method
        ---
        -> fuse.ScanSet.Unit
        """

    def make(self, key):
        self.insert1(key)
        pipe = (fuse.ScanDone() & key).fetch1('pipe')
        pipe = dj.create_virtual_module(pipe, 'pipeline_' + pipe)
        self.Unit().insert(
            fuse.ScanDone * pipe.ScanSet.Unit * pipe.MaskClassification.Type & key & dict(type='soma'),
            ignore_extra_fields=True)


@schema
class Preprocessing(dj.Lookup):
    definition = """
    # settings for movie preprocessing

    preproc_id       : tinyint # preprocessing ID
    ---
    resampl_freq     : decimal(3,1)  # resampling refrequency of stimuli and behavior
    behavior_lowpass : decimal(3,1)  # low pass cutoff of behavior signals Hz
    row              : tinyint # row size of movies
    col              : tinyint # col size of movie
    """

    @property
    def contents(self):
        yield from [[0, 30, 2.5, 36, 64], [1, 10, 2.5, 36, 64]]


@schema
class Tier(dj.Lookup):
    definition = """
    tier        : varchar(20)   # data tier
    ---
    """

    @property
    def contents(self):
        yield from zip(["train", "test", "validation"])


@schema
class ConditionTier(dj.Computed):
    definition = """
    # split into train, test, validation

    -> stimulus.Condition
    -> MovieScan
    ---
    -> Tier
    """

    @property
    def dataset_compositions(self):
        return dj.U('animal_id', 'session', 'scan_idx', 'stimulus_type', 'tier').aggr(
            self * stimulus.Condition(), n='count(*)')

    @property
    def key_source(self):
        return MovieScan() & stimulus.Trial()

    def check_train_test_split(self, clips, cond):
        stim = getattr(stimulus, cond['stimulus_type'].split('.')[-1])
        train_test = dj.U(*UNIQUE_CLIP[cond['stimulus_type']]).aggr(
            clips * stim, train='sum(1-test)', test='sum(test)') & 'train>0 and test>0'
        assert len(train_test) == 0, 'Train and test clips do overlap'

    def fill_up(self, tier, clips, cond, key, m):
        existing = ConditionTier().proj() & (self & dict(tier=tier)) \
            & (stimulus.Trial() * stimulus.Condition() & dict(key, **cond))
        n = len(existing)
        if n < m:
            # all hashes that are in clips but not registered for that animal and have the right tier
            candidates = dj.U('condition_hash') & (self & (dj.U('condition_hash') & (clips - self)) & dict(tier=tier))
            keys = candidates.fetch(dj.key)
            d = m - n
            update = min(len(keys), d)

            log.info('Inserting {} more existing {} trials'.format(update, tier))
            for k in keys[:update]:
                k = (clips & k).fetch1(dj.key)
                k['tier'] = tier
                self.insert1(k, ignore_extra_fields=True)

        existing = ConditionTier().proj() & (self & dict(tier=tier)) \
            & (stimulus.Trial() * stimulus.Condition() & dict(key, **cond))
        n = len(existing)
        if n < m:
            keys = (clips - self).fetch(dj.key)
            update = m - n
            log.info('Inserting {} more new {} trials'.format(update, tier))
            for k in keys[:update]:
                k['tier'] = tier
                self.insert1(k, ignore_extra_fields=True)

    def make(self, key):
        log.info('Processing ' + repr(key))
        conditions = dj.U('stimulus_type').aggr(stimulus.Condition() & (stimulus.Trial() & key), count='count(*)') \
            & 'stimulus_type in ("stimulus.Clip","stimulus.Monet", "stimulus.Monet2", "stimulus.Trippy", "stimulus.Matisse2")'
        for cond in conditions.fetch(as_dict=True):
            log.info('Checking condition {stimulus_type} (n={count})'.format(**cond))
            clips = (stimulus.Condition() * MovieScan() & key & cond).aggr(stimulus.Trial(), repeats="count(*)",
                                                                           test='count(*) > 4')
            self.check_train_test_split(clips, cond)

            m = len(clips)
            m_test = m_val = len(clips & 'test > 0') or max(m // 10, 1)
            log.info('Minimum test and validation set size will be {}'.format(m_test))

            # insert repeats as test trials
            self.insert((clips & dict(test=1)).proj(tier='"test"'), ignore_extra_fields=True)
            self.fill_up('test', clips, cond, key, m_test)
            self.fill_up('validation', clips, cond, key, m_val)
            self.fill_up('train', clips, cond, key, m - m_test - m_val)


@schema
class MovieClips(dj.Computed, FilterMixin):
    definition = """
    # movies subsampled

    -> stimulus.Condition
    -> Preprocessing
    ---
    fps0                 : float                # original framerate
    frames               : blob@scratch09       # input movie downsampled
    sample_times         : blob@scratch09       # sample times for the new frames
    duration             : float                # duration in seconds
    """

    @property
    def scan_keys(self):
        conditions = stimulus.Condition & (stimulus.Trial & MovieScanCandidate)
        return (conditions * Preprocessing & 'preproc_id=0').proj()

    def get_frame_rate(self, key):
        stimulus_type = (stimulus.Condition() & key).fetch1('stimulus_type')
        if stimulus_type == 'stimulus.Clip':
            assert len(stimulus.Clip() & key) == 1, 'key must specify exactly one clip'
            frame_rate = (stimulus.Movie() * stimulus.Clip() & key).fetch1('frame_rate')
        else:
            movie_rel = getattr(stimulus, stimulus_type.split('.')[-1])
            frame_rate = (movie_rel() & key).fetch1('fps')
        return float(frame_rate)  # in case it was a decimal

    def load_movie(self, key):
        # --- get correct stimulus relation
        log.info('Loading movie {condition_hash}'.format(**key))
        stimulus_type = (stimulus.Condition() & key).fetch1('stimulus_type')

        if stimulus_type == 'stimulus.Clip':
            assert len(stimulus.Clip() & key) == 1, 'key must specify exactly one clip'
            movie, frame_rate = (stimulus.Movie() * stimulus.Movie.Clip()
                                 * stimulus.Clip() & key).fetch1('clip', 'frame_rate')
            vid = imageio.get_reader(io.BytesIO(movie.tobytes()), 'ffmpeg')
            # convert to grayscale and stack to movie in width x height x time
            movie = np.stack([frame.mean(axis=-1) for frame in vid], axis=2)
        else:
            movie_rel = getattr(stimulus, stimulus_type.split('.')[-1])
            assert len(movie_rel() & key) == 1, 'key must specify exactly one clip'
            movie, frame_rate = (movie_rel() & key).fetch1('movie', 'fps')

        frame_rate = float(frame_rate)  # in case it was a decimal

        return movie, frame_rate

    def adjust_duration(self, key, base):
        if stimulus.Clip() & key:
            duration, skip_time = map(float, (stimulus.Clip() & key).fetch1('cut_after', 'skip_time'))
            duration = min(base.max(), duration)
            log.info('Stimulus duration is cut to {}s with {}s skiptime'.format(duration, skip_time))
        else:
            duration = base.max()
            skip_time = 0
            log.info('Stimulus duration is {}s (full length)'.format(duration))
        return duration, skip_time

    def make(self, key):
        log.info(80 * '-')
        log.info('Processing key ' + repr(key))
        sampling_period = 1 / float((Preprocessing & key).fetch1('resampl_freq'))
        imgsize = (Preprocessing() & key).fetch1('col', 'row')  # target size of movie frames

        log.info('Downsampling movie to {}'.format(repr(imgsize)))
        movie, frame_rate = self.load_movie(key)

        # --- downsampling movie
        h_movie = self.get_filter(sampling_period, 1 / frame_rate, 'hamming', warning=False)

        if not movie.shape[0] / imgsize[1] == movie.shape[1] / imgsize[0]:
            log.warning('Image size changes aspect ratio.')

        movie2 = np.stack([cv2.resize(m, imgsize, interpolation=cv2.INTER_AREA)
                           for m in movie.squeeze().transpose([2, 0, 1])], axis=0)
        movie = movie2.astype(np.float32).transpose([1, 2, 0])
        # low pass filter movie
        movie = np.apply_along_axis(lambda m: np.convolve(m, h_movie, mode='same'), axis=-1, arr=movie)
        base = np.arange(movie.shape[-1]) / frame_rate  # np.vstack([ft - ft[0] for ft in flip_times]).mean(axis=0)

        duration, skip_time = self.adjust_duration(key, base)
        samps = np.arange(0, duration, sampling_period)  # samps is relative to fliptime 0

        movie_spline = SplineMovie(base, movie, k=1, ext=1)
        movie = movie_spline(samps + skip_time).astype(np.float32)

        # --- generate response sampling points and sample movie frames relative to it
        self.insert1(dict(key, frames=movie.transpose([2, 0, 1]),
                          sample_times=samps, fps0=frame_rate, duration=duration))


@schema
class ResponseKeys(dj.Computed, TraceMixin):
    definition = """
    # response block keys
    -> MovieScan
    """

    class Unit(dj.Part):
        definition = """
        -> master
        -> fuse.ScanSet.Unit
        ---
        row_id           : int  # row id in the response block
        """

    def make(self, key):
        self.insert1(key)
        trace_keys = (fuse.ScanSet.Unit * MovieScan.Unit & key).fetch(
            dj.key, order_by='animal_id, session, scan_idx, unit_id')
        self.Unit().insert([dict(row_id=i, **k) for i, k in enumerate(trace_keys)])


@schema
class InputResponse(dj.Computed, FilterMixin, TraceMixin):
    definition = """
    # responses of one neuron to the stimulus

    -> MovieScan
    -> Preprocessing
    ---
    """

    key_source = MovieScan() * Preprocessing() & MovieClips()

    class Input(dj.Part):
        definition = """
            -> master
            -> stimulus.Trial
            -> MovieClips
            ---
            """

    class Response(dj.Part):
        definition = """
            -> master
            -> master.Input
            ---
            responses           : blob@scratch09    # reponse of one neurons for all bins
            """

    def get_trace_spline(self, key, sampling_period):
        traces, frame_times, trace_keys = self.load_traces_and_frametimes(key)
        log.info('Loaded {} traces'.format(len(traces)))
        median_frame_time = np.median(np.diff(frame_times))
        log.info('Median frame time: {}'.format(median_frame_time))
        h_trace = self.get_filter(sampling_period, median_frame_time, 'hamming', warning=False)
        log.info('Generating lowpass filters with cutoff {:.3f}Hz'.format(1 / sampling_period))
        # low pass filter
        trace_spline = SplineCurve(
            frame_times, [np.convolve(trace, h_trace, mode='same') for trace in traces], k=1, ext=1)
        return trace_spline, trace_keys, frame_times.min(), frame_times.max()

    def median_scan_period(self, key):
        frame_times = self.load_frame_times(key)
        return np.median(np.diff(frame_times)).item()

    def make(self, scan_key):
        log.info(80 * '-')
        log.info('Populating {}'.format(repr(scan_key)).ljust(80, '-'))
        self.insert1(scan_key)
        # integration window size for responses
        sampling_period = 1 / float((Preprocessing & scan_key).fetch1('resampl_freq'))

        log.info('Sampling neural responses at {}s intervals'.format(sampling_period))
        trace_spline, _, ftmin, ftmax = self.get_trace_spline(scan_key, sampling_period)

        assert ConditionTier & scan_key, 'ConditionTier has not been populated'

        conditions = stimulus.Condition & (stimulus.Trial & scan_key)
        clips = (conditions * Preprocessing & scan_key).proj()
        assert not clips - MovieClips, 'MovieClips has not been populated'

        flip_times, sample_times, fps0, trial_keys = (MovieScan() * MovieClips() * stimulus.Trial() & scan_key).fetch(
            'flip_times', 'sample_times', 'fps0', dj.key)
        flip_times = [ft.squeeze() for ft in flip_times]
        nodrop = np.array([np.diff(ft).max() < 1.99 / frame_rate for ft, frame_rate in zip(flip_times, fps0)])
        valid = np.array([ft.min() >= ftmin and ft.max() <= ftmax for ft in flip_times], dtype=bool)
        if not np.all(nodrop & valid):
            log.warning('Dropping {} trials with dropped frames or flips outside the recording interval'.format(
                (~(nodrop & valid)).sum()))
        for trial_key, flips, samps, take in tqdm(zip(trial_keys, flip_times, sample_times, nodrop & valid),
                                                  total=len(trial_keys), desc='Trial '):
            if take:
                self.Input().insert1(trial_key)
                self.Response().insert1(dict(responses=trace_spline(flips[0] + samps), **trial_key))


@schema
class Eye(dj.Computed, FilterMixin, BehaviorMixin):
    definition = """
    # eye movement data

    -> InputResponse.Input
    ---
    -> pupil.FittedPupil
    pupil              : blob@scratch09 # pupil dilation trace
    dpupil             : blob@scratch09 # derivative of pupil dilation trace
    center             : blob@scratch09 # center position of the eye
    """

    @property
    def key_source(self):
        return InputResponse & pupil.FittedPupil & stimulus.BehaviorSync

    def make(self, scan_key):
        # pick out the "latest" tracking method to use for pupil info extraction
        scan_key['tracking_method'] = (pupil.FittedPupil & scan_key).fetch(
            'tracking_method', order_by='tracking_method')[-1]
        log.info('Populating\n' + pformat(scan_key, indent=10))
        radius, xy, eye_time = self.load_eye_traces(scan_key)
        frame_times = InputResponse().load_frame_times(scan_key)
        behavior_clock = self.load_behavior_timing(scan_key)

        if len(frame_times) - len(behavior_clock) != 0:
            assert abs(len(frame_times) - len(behavior_clock)) < 2, 'Difference bigger than 2 time points'
            l = min(len(frame_times), len(behavior_clock))
            log.info('Frametimes and stimulus.BehaviorSync differ in length! Shortening it.', depth=1)
            frame_times = frame_times[:l]
            behavior_clock = behavior_clock[:l]

        fr2beh = NaNSpline(frame_times, behavior_clock, k=1, ext=3)
        sampling_period = 1 / float((Preprocessing & scan_key).fetch1('behavior_lowpass'))
        log.info('Downsampling eye signal to {}Hz'.format(1 / sampling_period))
        deye = np.nanmedian(np.diff(eye_time))
        h_eye = self.get_filter(sampling_period, deye, 'hamming', warning=True)
        h_deye = self.get_filter(sampling_period, deye, 'dhamming', warning=True)
        pupil_spline = NaNSpline(
            eye_time, np.convolve(radius, h_eye, mode='same'), k=1, ext=0)
        dpupil_spline = NaNSpline(
            eye_time, np.convolve(radius, h_deye, mode='same'), k=1, ext=0)
        center_spline = SplineCurve(
            eye_time, np.vstack([np.convolve(coord, h_eye, mode='same') for coord in xy]), k=1, ext=0)

        flip_times, sample_times, trial_keys = (InputResponse.Input() * MovieClips() * stimulus.Trial() & scan_key).fetch(
            'flip_times', 'sample_times', dj.key)
        flip_times = [ft.squeeze() for ft in flip_times]
        for trial_key, flips, samps in tqdm(zip(trial_keys, flip_times, sample_times),
                                            total=len(trial_keys), desc='Trial '):
            t = fr2beh(flips[0] + samps)
            pupil_trace = pupil_spline(t)
            dpupil = dpupil_spline(t)
            center = center_spline(t)
            nans = np.array([np.isnan(e).sum() for e in [pupil_trace, dpupil, center]])
            if np.any(nans > 0):
                log.info('Found {} NaNs in one of the traces. Skipping trial {}'.format(
                    np.max(nans), pformat(trial_key, indent=5)))
            else:
                self.insert1(dict(scan_key, **trial_key,
                                  pupil=pupil_trace,
                                  dpupil=dpupil,
                                  center=center),
                             ignore_extra_fields=True)


@schema
class Eye2(dj.Computed, FilterMixin, BehaviorMixin):
    definition = """
    # eye movement data

    -> InputResponse.Input
    -> pupil.FittedContour
    ---
    pupil              : blob@scratch09 # pupil dilation trace
    dpupil             : blob@scratch09 # derivative of pupil dilation trace
    center             : blob@scratch09 # center position of the eye
    """

    @property
    def key_source(self):
        return InputResponse & pupil.FittedContour & stimulus.BehaviorSync

    def make(self, scan_key):
        log.info('Populating\n' + pformat(scan_key, indent=10))
        radius, xy, eye_time = self.load_eye_traces_old(scan_key)
        frame_times = InputResponse().load_frame_times(scan_key)
        behavior_clock = self.load_behavior_timing(scan_key)

        if len(frame_times) - len(behavior_clock) != 0:
            assert abs(len(frame_times) - len(behavior_clock)) < 2, 'Difference bigger than 2 time points'
            l = min(len(frame_times), len(behavior_clock))
            log.info('Frametimes and stimulus.BehaviorSync differ in length! Shortening it.', depth=1)
            frame_times = frame_times[:l]
            behavior_clock = behavior_clock[:l]

        fr2beh = NaNSpline(frame_times, behavior_clock, k=1, ext=3)
        sampling_period = 1 / float((Preprocessing & scan_key).fetch1('behavior_lowpass'))
        log.info('Downsampling eye signal to {}Hz'.format(1 / sampling_period))
        deye = np.nanmedian(np.diff(eye_time))
        h_eye = self.get_filter(sampling_period, deye, 'hamming', warning=True)
        h_deye = self.get_filter(sampling_period, deye, 'dhamming', warning=True)
        pupil_spline = NaNSpline(
            eye_time, np.convolve(radius, h_eye, mode='same'), k=1, ext=0)
        dpupil_spline = NaNSpline(
            eye_time, np.convolve(radius, h_deye, mode='same'), k=1, ext=0)
        center_spline = SplineCurve(
            eye_time, np.vstack([np.convolve(coord, h_eye, mode='same') for coord in xy]), k=1, ext=0)

        flip_times, sample_times, trial_keys = (InputResponse.Input() * MovieClips() * stimulus.Trial() & scan_key).fetch(
            'flip_times', 'sample_times', dj.key)
        flip_times = [ft.squeeze() for ft in flip_times]
        for trial_key, flips, samps in tqdm(zip(trial_keys, flip_times, sample_times),
                                            total=len(trial_keys), desc='Trial '):
            t = fr2beh(flips[0] + samps)
            pupil = pupil_spline(t)
            dpupil = dpupil_spline(t)
            center = center_spline(t)
            nans = np.array([np.isnan(e).sum() for e in [pupil, dpupil, center]])
            if np.any(nans > 0):
                log.info('Found {} NaNs in one of the traces. Skipping trial {}'.format(
                    np.max(nans), pformat(trial_key, indent=5)))
            else:
                self.insert1(dict(scan_key, **trial_key,
                                  pupil=pupil,
                                  dpupil=dpupil,
                                  center=center),
                             ignore_extra_fields=True)


@schema
class Treadmill(dj.Computed, FilterMixin, BehaviorMixin):
    definition = """
    # treadmill data

    -> InputResponse.Input
    -> treadmill.Treadmill
    ---
    treadmill          : blob@scratch09 # treadmill speed (|velcolity|)
    """

    @property
    def key_source(self):
        rel = InputResponse
        return rel & treadmill.Treadmill() & stimulus.BehaviorSync()

    def make(self, scan_key):
        print('Populating', pformat(scan_key))
        v, treadmill_time = self.load_treadmill_velocity(scan_key)
        frame_times = InputResponse().load_frame_times(scan_key)
        behavior_clock = self.load_behavior_timing(scan_key)

        if len(frame_times) - len(behavior_clock) != 0:
            assert abs(len(frame_times) - len(behavior_clock)) < 2, 'Difference bigger than 2 time points'
            l = min(len(frame_times), len(behavior_clock))
            log.info('Frametimes and stimulus.BehaviorSync differ in length! Shortening it.')
            frame_times = frame_times[:l]
            behavior_clock = behavior_clock[:l]

        fr2beh = NaNSpline(frame_times, behavior_clock, k=1, ext=3)
        sampling_period = 1 / float((Preprocessing & scan_key).fetch1('behavior_lowpass'))
        log.info('Downsampling treadmill signal to {}Hz'.format(1 / sampling_period))

        h_tread = self.get_filter(sampling_period, np.nanmedian(np.diff(treadmill_time)), 'hamming', warning=True)
        treadmill_spline = NaNSpline(treadmill_time, np.abs(np.convolve(v, h_tread, mode='same')), k=1, ext=0)

        flip_times, sample_times, trial_keys = (InputResponse.Input() * MovieClips() * stimulus.Trial() & scan_key).fetch(
            'flip_times', 'sample_times', dj.key)
        flip_times = [ft.squeeze() for ft in flip_times]
        for trial_key, flips, samps in tqdm(zip(trial_keys, flip_times, sample_times),
                                            total=len(trial_keys), desc='Trial '):
            tm = treadmill_spline(fr2beh(flips[0] + samps))
            nans = np.isnan(tm)
            if np.any(nans):
                log.info('Found {} NaNs in one of the traces. Skipping trial {}'.format(
                    nans.sum(), pformat(trial_key, indent=5)))
            else:
                self.insert1(dict(scan_key, **trial_key, treadmill=tm),
                             ignore_extra_fields=True)


@schema
class ScanDataset(dj.Computed):
    definition = """
    # scan hdf5 dataset

    -> InputResponse
    ---
    h5_dataset      : attach@scratch09  # hdf5 dataset
    """

    def make(self, key):
        assert ResponseKeys & key, 'ResponseKeys has not been populated'
        assert Eye & key or Eye2 & key, 'Eye/Eye2 has not been populated'
        assert Treadmill & key, 'Treadmill has not been populated'
        fdir = os.path.join('/tmp', self.database, self.table_name, key_hash(key))
        if not os.path.exists(fdir):
            os.makedirs(fdir)
        fpath = os.path.join(fdir, 'scanset.h5')
        save_dict_to_hdf5(self.compute_data(key), fpath)
        self.insert1(dict(h5_dataset=fpath, **key))

    def units(self, key=None):
        key = self.fetch1(dj.key) if key is None else (InputResponse & key).fetch1(dj.key)
        animal_ids, sessions, scan_idx, pipe_versions, segmentation_methods, spike_methods, unit_ids = \
            (ResponseKeys.Unit & key).fetch('animal_id', 'session', 'scan_idx', 'pipe_version',
                                            'segmentation_method', 'spike_method', 'unit_id', order_by='row_id')
        units = dict(
            animal_ids=animal_ids.astype(np.uint16),
            sessions=sessions.astype(np.uint8),
            scan_idx=scan_idx.astype(np.uint8),
            pipe_versions=pipe_versions.astype(np.uint8),
            segmentation_methods=segmentation_methods.astype(np.uint8),
            spike_methods=spike_methods.astype(np.uint8),
            unit_ids=unit_ids.astype(np.uint16)
        )
        return units

    def eye_table(self, key=None):
        # patch to deal with old eye tracking method
        key = self.fetch1(dj.key) if key is None else (InputResponse & key).fetch1(dj.key)
        return Eye if Eye & key else Eye2

    def compute_data(self, key=None):
        key = self.fetch1(dj.key) if key is None else (InputResponse & key).fetch1(dj.key)
        log.info('Computing dataset for {}'.format(repr(key)))

        # get neurons
        neurons = self.units(key)
        assert len(np.unique(neurons['unit_ids'])) == len(neurons['unit_ids']), \
            'unit ids are not unique, do you have more than one preprocessing method?'

        # get data relation
        data_rel = InputResponse.Input * InputResponse.Response * stimulus.Condition.proj('stimulus_type') \
            * MovieClips * ConditionTier & key
        EyeTable = self.eye_table(key)
        include_behavior = bool(EyeTable * Treadmill & key)
        if include_behavior:
            # restrict trials to those that do not have NaNs in Treadmill or Eye
            data_rel = data_rel & EyeTable & Treadmill

        # --- fetch all stimuli and classify into train/test/val
        inputs, hashes, stim_keys, tiers, types, trial_idx, durations = data_rel.fetch(
            'frames', 'condition_hash', dj.key, 'tier', 'stimulus_type', 'trial_idx', 'duration',
            order_by='condition_hash ASC, trial_idx ASC')
        train_idx = np.array([t == 'train' for t in tiers], dtype=bool)
        test_idx = np.array([t == 'test' for t in tiers], dtype=bool)
        val_idx = np.array([t == 'validation' for t in tiers], dtype=bool)

        # ----- extract trials

        responses, behavior, eye_position = [], [], []
        for stim_key in tqdm(stim_keys):
            response_block = (InputResponse.Response & stim_key).fetch1('responses')
            responses.append(response_block.T.astype(np.float32))
            if include_behavior:
                pupil, dpupil, treadmill, center = (EyeTable * Treadmill & stim_key).fetch1(
                    'pupil', 'dpupil', 'treadmill', 'center')
                behavior.append(np.vstack([pupil, dpupil, treadmill]).T)
                eye_position.append(center.T)

        # insert channel dimension
        for i, inp in enumerate(inputs):
            if len(inp.shape) == 3:
                inputs[i] = inp[None, ...]

        hashes = hashes.astype(str)
        types = types.astype(str)

        def run_stats(selector, types, ix, axis=None):
            ret = {}
            for t in np.unique(types):
                train_responses = selector(ix & (types == t))
                ret[t] = dict(
                    mean=train_responses.mean(axis=axis).astype(np.float32),
                    std=train_responses.std(axis=axis, ddof=1).astype(np.float32),
                    min=train_responses.min(axis=axis).astype(np.float32),
                    max=train_responses.max(axis=axis).astype(np.float32),
                    median=np.median(train_responses, axis=axis).astype(np.float32)
                )
            train_responses = selector(ix)
            ret['all'] = dict(
                mean=train_responses.mean(axis=axis).astype(np.float32),
                std=train_responses.std(axis=axis, ddof=1).astype(np.float32),
                min=train_responses.min(axis=axis).astype(np.float32),
                max=train_responses.max(axis=axis).astype(np.float32),
                median=np.median(train_responses, axis=axis).astype(np.float32)
            )
            return ret

        # --- compute statistics
        log.info('Computing statistics on training dataset')
        def response_selector(ix): return np.concatenate([r for take, r in zip(ix, responses) if take], axis=0)
        response_statistics = run_stats(response_selector, types, train_idx, axis=0)

        def input_selector(ix): return np.hstack([r.ravel() for take, r in zip(ix, inputs) if take])
        input_statistics = run_stats(input_selector, types, train_idx)

        statistics = dict(
            inputs=input_statistics,
            responses=response_statistics
        )

        if include_behavior:
            # ---- include statistics
            def behavior_selector(ix): return np.concatenate([r for take, r in zip(ix, behavior) if take], axis=0)
            behavior_statistics = run_stats(behavior_selector, types, train_idx, axis=0)

            def eye_selector(ix): return np.concatenate([r for take, r in zip(ix, eye_position) if take], axis=0)
            eye_statistics = run_stats(eye_selector, types, train_idx, axis=0)

            statistics['behavior'] = behavior_statistics
            statistics['eye_position'] = eye_statistics

        retval = dict(inputs=inputs,
                      responses=responses,
                      types=types.astype('S'),
                      train_idx=train_idx,
                      val_idx=val_idx,
                      test_idx=test_idx,
                      condition_hashes=hashes.astype('S'),
                      durations=durations.astype(np.float32),
                      trial_idx=trial_idx.astype(np.uint32),
                      neurons=neurons,
                      tiers=tiers.astype('S'),
                      statistics=statistics
                      )
        if include_behavior:
            retval['behavior'] = behavior
            retval['eye_position'] = eye_position
        return retval

    def valid_trials(self, key=None):
        key = self.fetch(dj.key) if key is None else (self & key).fetch(dj.key)
        responses = InputResponse.Response & key
        treadmills = Treadmill & key
        eyes = (Eye & key) or (Eye2 & key)
        return (responses * treadmills * eyes).proj()


schema.spawn_missing_classes()
