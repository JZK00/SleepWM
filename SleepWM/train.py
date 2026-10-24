"""Choose staged training from a reference model or joint supervised training."""
import argparse
import sys


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--mode', choices=('staged', 'supervised'), default='supervised')
    selected, remaining = parser.parse_known_args(values)
    if not values or values == ['--help'] or values == ['-h']:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--mode', choices=('staged', 'supervised'), default='supervised',
                            help='staged: continue from a reference model; supervised: joint training from scratch')
        parser.print_help()
        print('\nUse python train.py --mode staged --help or --mode supervised --help for options.')
        return
    if selected.mode == 'staged':
        from sleepwm.staged import main as run
    else:
        from sleepwm.supervised import main as run
    return run(remaining)


if __name__ == '__main__':
    main()
