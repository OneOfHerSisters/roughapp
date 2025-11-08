import sys
from roughapp.cli.commands import build_parser, cmd_info, cmd_introduce


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    
    if args.cmd == "info":
        cmd_info(args)
    elif args.cmd == "introduce":
        cmd_introduce(args)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()