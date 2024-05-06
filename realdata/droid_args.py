
def add_droid_argparse(parser):
    # parser.add_argument("--imagedir", type=str, help="path to image directory")
    # parser.add_argument("--pcd_path", type=str, help="path save pointcloud data", default="")

    # parser.add_argument("--calib", type=str, help="path to calibration file")

    parser.add_argument("--weights", default="../libraries/DROID-SLAM/droid.pth")
    parser.add_argument("--buffer", type=int, default=1024)
    parser.add_argument("--image_size", default=[240, 320], nargs=2, type=int)
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")

    return parser