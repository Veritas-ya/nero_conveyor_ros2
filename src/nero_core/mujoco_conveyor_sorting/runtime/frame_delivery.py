"""Bounded ordered delivery and counters; device frames are never synthesized."""
from collections import deque


class FrameDelivery:
    def __init__(self, capacity=8):
        self.frames=deque(maxlen=capacity)
        self.received=self.overflow=self.stale=self.delivered=0
        self.streams={}
        self.ordered_reader_active=False

    def append(self,frame):
        if len(self.frames)==self.frames.maxlen:self.overflow+=1
        self.frames.append(frame);self.received+=1
        for name in ('depth','color'):
            timing=(frame.timestamp_metadata or {}).get(name) or {}
            stamp,index=timing.get('device_time_s'),timing.get('frame_index')
            row=self.streams.setdefault(name,{'frames':0,'missing_device_indices':0,
                'resets':0,'first_s':None,'last_s':None,'last_index':None,'max_gap_s':0.})
            if stamp is None:continue
            if row['last_s'] is not None:
                gap=stamp-row['last_s'];row['max_gap_s']=max(row['max_gap_s'],gap)
                if gap<0 or (index is not None and row['last_index'] is not None and index<row['last_index']):
                    row['resets']+=1
                elif index is not None and row['last_index'] is not None:
                    row['missing_device_indices']+=max(0,index-row['last_index']-1)
            if row['first_s'] is None:row['first_s']=stamp
            row.update(last_s=stamp,last_index=index,frames=row['frames']+1)

    def take(self,after_sequence,now,max_age):
        self.ordered_reader_active=True
        while self.frames:
            frame=self.frames[0]
            if after_sequence is not None and frame.header.sequence<=after_sequence:
                self.frames.popleft();continue
            if frame.header.age(now)>max_age:
                self.frames.popleft();self.stale+=1;continue
            self.delivered+=1
            return self.frames.popleft()

    def report(self):
        streams={}
        for name,r in self.streams.items():
            dt=0 if r['first_s'] is None else r['last_s']-r['first_s']
            streams[name]={**r,'received_fps':(r['frames']-1)/dt if dt>0 and not r['resets'] else None}
        return {'received_frames':self.received,'ordered_delivered_frames':self.delivered,
            'ordered_reader_active':self.ordered_reader_active,
            'buffer_overflow_frames':self.overflow,'stale_discarded_frames':self.stale,
            'buffered_frames':len(self.frames),'buffer_capacity':self.frames.maxlen,'streams':streams}
