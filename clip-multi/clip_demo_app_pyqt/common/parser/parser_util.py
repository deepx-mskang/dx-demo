import argparse

class ParserUtil:
    @staticmethod
    def get_args():
        # fmt: off
        parser = argparse.ArgumentParser(description="Generate Similarity Matrix from ONNX files")

        # Dataset Path Arguments
        parser.add_argument("--base_path", type=str, default="assets", help="Videos directory")

        # Legacy UI arguments still referenced by SettingsView.
        parser.add_argument("--max_words", type=int, default=32, help="")
        parser.add_argument("--feature_framerate", type=int, default=1, help="")
        parser.add_argument("--slice_framepos", type=int, default=0, choices=[0, 1, 2],
                            help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")

        # App Configuration Argument
        parser.add_argument("--number_of_channels", "--stream", dest="number_of_channels", type=int, default=16,
                            help="Number of input video channels")
        parser.add_argument("--settings_mode", type=int, default=0, help="Settings Mode Setting (off: 0, on: 1)")
        parser.add_argument("--camera_mode", type=int, default=0, help="Camera Mode Setting (off: 0, on: 1)")
        parser.add_argument("--camera", dest="camera_mode", action="store_const", const=1,
                            help="Enable camera mode")
        parser.add_argument("--merge_central_grid", type=int, default=0,
                            help="Merge the Centeral Grid Setting (off: 0, on: 1)")
        parser.add_argument("--video_fps_sync_mode", type=int, default=0, help="Video FPS Sync Mode Setting (off: 0, on: 1)")
        parser.add_argument("--skip_settings", action="store_true",
                            help="Skip the settings window and launch directly")

        # DXNN Inference Engine Configuration Argument
        parser.add_argument("--inference_engine_async_mode", type=int, default=1, help="Inference Engine RunAsync Mode (off: 0, on: 1)")

        # Model Path Arguments
        parser.add_argument("--text_encoder_onnx", type=str,
                            default="../clip-single/onnx/ViT-L-14-quickgelu-dfn2b-text.onnx",
                            help="ONNX file path for text encoder")
        parser.add_argument("--video_encoder_dxnn", type=str,
                            default="../clip-single/dxnn/ViT-L-14-quickgelu-dfn2b.dxnn",
                            help="DXNN file path for video encoder")

        # Direct UI settings
        parser.add_argument("--show_percent", type=int, choices=[0, 1], default=0, help="Display percentage (off: 0, on: 1)")
        parser.add_argument("--show_score", type=int, choices=[0, 1], default=0, help="Display score (off: 0, on: 1)")
        parser.add_argument("--fullscreen_mode", type=int, choices=[0, 1], default=1, help="Fullscreen mode (off: 0, on: 1)")
        parser.add_argument("--dark_theme", type=int, choices=[0, 1], default=1, help="Dark theme (off: 0, on: 1)")
        parser.add_argument("--show_each_fps_label", type=int, choices=[0, 1], default=0,
                            help="Display FPS for each video (off: 0, on: 1)")
        parser.add_argument("--dynamic_font_mode", type=str, default="fit",
                            help="Text layout mode")
        parser.add_argument("--min_font_size", type=int, default=13,
                            help="Minimum font size")

        return parser.parse_args()
