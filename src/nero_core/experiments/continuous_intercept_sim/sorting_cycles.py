"""Session bookkeeping: existing return completion requests original factory feed."""
class SortingCycles:
    def __init__(self,count):
        if not isinstance(count,int) or count<1:raise ValueError('cycles must be positive')
        self.count=count
        self.previous_stage=None
        self.previous_target=None
        self.returns=[]
        self.feed_requests=0

    def observe(self,stage,target,now,request_next,*,reason):
        if (self.previous_stage=='returning' and stage=='waiting'
                and self.previous_target is not None and reason=='wait_pose_reached'):
            self.returns.append(dict(object_id=self.previous_target,returned_at_s=float(now)))
            if len(self.returns)<self.count:
                request_next()
                self.feed_requests+=1
        self.previous_stage=stage
        self.previous_target=target
        return len(self.returns)>=self.count
