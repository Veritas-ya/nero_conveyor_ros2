"""Cooperative CPU slices for completed-record serialization, never control.

A background Python thread still shares the GIL with the CAN reader. Bound
pure-Python conversion/encoding slices and give that reader time to drain.
"""
import json
import time


class ArchiveWork:
    def __init__(self, *, work_s=.001, yield_s=.001, clock=time.monotonic, sleep=time.sleep):
        self.work_s, self.yield_s = work_s, yield_s
        self.clock, self.sleep = clock, sleep
        self.since = clock()
        self.yields = 0
        self.yield_wall_s = 0.

    def checkpoint(self):
        now = self.clock()
        if now-self.since >= self.work_s:
            self.sleep(self.yield_s)
            resumed = self.clock()
            self.yield_wall_s += resumed-now
            self.yields += 1
            self.since = resumed

    def rows(self, values):
        for value in values:
            self.checkpoint()
            yield value

    def write_json(self, path, value):
        # dumps(indent=2) materializes hundreds of thousands of fragments
        # before joining them. Stream bounded chunks with explicit yields.
        encoder = json.JSONEncoder(ensure_ascii=False, indent=2)
        with path.open('w', encoding='utf-8') as stream:
            chunks, size = [], 0
            for chunk in encoder.iterencode(value):
                chunks.append(chunk)
                size += len(chunk)
                if size >= 32768:
                    stream.write(''.join(chunks))
                    chunks, size = [], 0
                self.checkpoint()
            stream.write(''.join(chunks)+'\n')

    def diagnostics(self):
        return dict(mode='cooperative_thread', work_slice_s=self.work_s,
                    yield_s=self.yield_s, yields=self.yields, yield_wall_s=self.yield_wall_s)
