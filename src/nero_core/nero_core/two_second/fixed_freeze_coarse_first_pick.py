"""Independent fixed-freeze picking entry: coarse A, coarse B, eligible dense retries."""
from .main import main as hardware_main


def main(argv=None):
    return hardware_main(argv, freeze_source='single_exposure_native_depth',
                         candidate_retry_policy='coarse_first')


if __name__ == '__main__':
    raise SystemExit(main())
