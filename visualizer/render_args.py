# default argparse options for connecting visualizer

def add_render_argparse(parser):
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument(
        "--thickness", type=float, default=0.01,
        help="Thickness (radius) of character body"
    )
    #################### render options #################################
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--axis-up", type=str, choices=["x", "y", "z"], default="y"
    )
    parser.add_argument(
        "--axis-face", type=str, choices=["x", "y", "z"], default="z"
    )
    parser.add_argument(
        "--camera-position",
        nargs="+",
        type=float,
        required=False,
        default=(2.0, 2.0, 2.0),
    )
    parser.add_argument(
        "--camera-origin",
        nargs="+",
        type=float,
        required=False,
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument("--hide-origin", action="store_true")
    parser.add_argument("--render-overlay", action="store_true")
    parser.add_argument(
        "--x-offset",
        type=int,
        default=2,
        help="Translates each character by x-offset*idx to display them "
        "simultaneously side-by-side",
    )
    return parser