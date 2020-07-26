from functools import partial
import datajoint as dj
import numpy as np

from neuro_data.utils.data import fill_nans
from .schema_bridge import *
from .. import logger as log


class TraceMixin:
    def load_frame_times(self, key):
        pipe = (fuse.Activity() & key).fetch('pipe')
        assert len(np.unique(pipe)) == 1, 'Selection is from different pipelines'
        pipe = dj.create_virtual_module(pipe[0], 'pipeline_' + pipe[0])
        k = dict(key)
        k.pop('field', None)
        ndepth = len(dj.U('z') & (pipe.ScanInfo.Field() & k))
        return (stimulus.Sync() & key).fetch1('frame_times').squeeze()[::ndepth]

    def load_traces_and_frametimes(self, key):
        from .data_schemas import MovieScan
        # -- find number of recording depths
        pipe = (fuse.Activity() & key).fetch('pipe')
        assert len(np.unique(pipe)) == 1, 'Selection is from different pipelines'
        pipe = dj.create_virtual_module(pipe[0], 'pipeline_' + pipe[0])
        k = dict(key)
        k.pop('field', None)
        ndepth = len(dj.U('z') & (pipe.ScanInfo.Field() & k))
        frame_times = (stimulus.Sync() & key).fetch1('frame_times').squeeze()[::ndepth]

        soma = pipe.MaskClassification.Type() & dict(type='soma')

        spikes = (dj.U('field', 'channel') * pipe.Activity.Trace() * MovieScan.Unit()
                  * pipe.ScanSet.UnitInfo() & soma & key)
        traces, ms_delay, trace_keys = spikes.fetch('trace', 'ms_delay', dj.key,
                                                    order_by='animal_id, session, scan_idx, unit_id')
        delay = np.fromiter(ms_delay / 1000, dtype=np.float)
        frame_times = (delay[:, None] + frame_times[None, :])
        traces = np.vstack([fill_nans(tr.astype(np.float32)).squeeze() for tr in traces])
        traces, frame_times = self.adjust_trace_len(traces, frame_times)
        return traces, frame_times, trace_keys

    def adjust_trace_len(self, traces, frame_times):
        trace_len, nframes = traces.shape[1], frame_times.shape[1]
        if trace_len < nframes:
            frame_times = frame_times[:, :trace_len]
        elif trace_len > nframes:
            traces = traces[:, :nframes]
        return traces, frame_times


class BehaviorMixin:
    def load_eye_traces_old(self, key):
        """
        Older method for loading eye traces, using FittedContour. Explicitly used by Eye2
        """
        r, center = (pupil.FittedContour.Ellipse() & key).fetch('major_r', 'center', order_by='frame_id ASC')
        detectedFrames = ~np.isnan(r)
        xy = np.full((len(r), 2), np.nan)
        xy[detectedFrames, :] = np.vstack(center[detectedFrames])
        xy = np.vstack(list(map(partial(fill_nans, preserve_gap=3), xy.T)))
        if np.any(np.isnan(xy)):
            log.info('Keeping some nans in the pupil location trace')
        pupil_radius = fill_nans(r.squeeze(), preserve_gap=3)
        if np.any(np.isnan(pupil_radius)):
            log.info('Keeping some nans in the pupil radius trace')

        eye_time = (pupil.Eye() & key).fetch1('eye_time').squeeze()
        return pupil_radius, xy, eye_time

    def load_eye_traces(self, key):
        r, center = (pupil.FittedPupil.Circle() & key).fetch('radius', 'center', order_by='frame_id')
        #r, center = (pupil.FittedContour.Ellipse() & key).fetch('major_r', 'center', order_by='frame_id ASC')
        detectedFrames = ~np.isnan(r)
        xy = np.full((len(r), 2), np.nan)
        xy[detectedFrames, :] = np.vstack(center[detectedFrames])
        xy = np.vstack(list(map(partial(fill_nans, preserve_gap=3), xy.T)))
        if np.any(np.isnan(xy)):
            log.info('Keeping some nans in the pupil location trace')
        pupil_radius = fill_nans(r.squeeze(), preserve_gap=3)
        if np.any(np.isnan(pupil_radius)):
            log.info('Keeping some nans in the pupil radius trace')

        eye_time = (pupil.Eye() & key).fetch1('eye_time').squeeze()
        return pupil_radius, xy, eye_time

    def load_behavior_timing(self, key):
        log.info('Loading behavior frametimes')
        # -- find number of recording depths
        pipe = (fuse.Activity() & key).fetch('pipe')
        assert len(np.unique(pipe)) == 1, 'Selection is from different pipelines'
        pipe = dj.create_virtual_module(pipe[0], 'pipeline_' + pipe[0])
        k = dict(key)
        k.pop('field', None)
        ndepth = len(dj.U('z') & (pipe.ScanInfo.Field() & k))
        return (stimulus.BehaviorSync() & key).fetch1('frame_times').squeeze()[0::ndepth]

    def load_treadmill_velocity(self, key):
        t, v = (treadmill.Treadmill() & key).fetch1('treadmill_time', 'treadmill_vel')
        return v.squeeze(), t.squeeze()
