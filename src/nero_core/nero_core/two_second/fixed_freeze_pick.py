"""Independent hardware entry: single-frame clouds, unchanged pick executor."""
from .main import main as hardware_main


def main(argv=None):
    return hardware_main(argv,freeze_source='single_exposure_native_depth')


if __name__=='__main__':raise SystemExit(main())
