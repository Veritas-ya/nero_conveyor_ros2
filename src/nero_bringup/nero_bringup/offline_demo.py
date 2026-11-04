"""Explicitly simulated orchestration example; no motion planning or physical model."""
from nero_core.task import TaskGate

def run():
    task = TaskGate()
    print('demo ONLY: synthetic detection, ideal acknowledgements, no robot or physics')
    task.start('demo-1', 'synthetic-target-1')
    for event in ('certified','at_follow','grasp_confirmed','lifted','released','home'):
        task.advance(event)
        print(event, '->', task.stage.value)
    print('ready for next distinct target')

def main(): run()
if __name__ == '__main__': main()
