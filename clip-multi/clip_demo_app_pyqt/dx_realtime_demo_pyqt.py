import sys
import os
import subprocess

# Added temporary workaround for PyQt and dx_engine conflict on Windows
if os.name == "nt":
    import onnxruntime
    from dx_engine import InferenceEngine

from PyQt5.QtWidgets import QApplication
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from clip_demo_app_pyqt.common.parser.parser_util import ParserUtil

from PyQt5.QtCore import pyqtSignal, QObject
from clip_demo_app_pyqt.view.settings_view import SettingsView
from PyQt5.QtGui import QFontDatabase, QFont

def is_vaapi_available():
    try:
        result = subprocess.run(
            ["gst-inspect-1.0", "vaapidecodebin"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False

use_vaapi = False
if os.name != "nt":
    if is_vaapi_available():
        use_vaapi = True
        sys.path.insert(0, "/usr/lib/python3/dist-packages")
        print("VA-API detected, path added.")

if os.name == "nt":
    import ctypes
    for p in os.environ.get("PATH").split(";"):
        dxrtlib = os.path.join(p, "dxrt.dll")
        if os.path.exists(dxrtlib):
            ctypes.windll.LoadLibrary(dxrtlib)

settings_ctx_ref = None

# fmt: on
class UIThread(QObject):
    run_ui_signal = pyqtSignal(SettingsView)
    
    def __init__(self):
        super().__init__()
        
    def run_ui(self, settings_ctx):
        self.run_ui_signal.emit(settings_ctx)

ui_thread = UIThread()

def success_cb(settings_ctx: SettingsView):
    ui_thread.run_ui_signal.connect(lambda ctx: start_main_ui(ctx))
    ui_thread.run_ui(settings_ctx)
    

def start_main_ui(settings_ctx: SettingsView):
    from clip_demo_app_pyqt.model.clip_model import ClipModel
    from clip_demo_app_pyqt.view.clip_view import ClipView
    from clip_demo_app_pyqt.viewmodel.clip_view_model import ClipViewModel
    settings_ctx.model = ClipModel(settings_ctx.base_path, settings_ctx.adjusted_video_path_lists,
                                    settings_ctx.sentence_list)
    settings_ctx.view_model = ClipViewModel(settings_ctx.model)
    settings_ctx.main_app = ClipView(settings_ctx.view_model, settings_ctx.ui_config,
                                        settings_ctx.base_path, settings_ctx.adjusted_video_path_lists,
                                        settings_ctx.merged_video_grid_info)
    global settings_ctx_ref
    settings_ctx_ref = settings_ctx
    settings_ctx.main_app.show()
    

def main():

    if "XDG_SESSION_TYPE" in os.environ:
        os.environ["XDG_SESSION_TYPE"] = "xcb"

    # Get Input Arguments
    args = ParserUtil.get_args()

    app = QApplication(sys.argv)
    
    app_font = QFontDatabase.addApplicationFont("./install_dep/DejaVuSans.ttf")
    font_families = QFontDatabase.applicationFontFamilies(app_font)
    
    settings_window = SettingsView(args, success_cb)
    settings_window.setFont(QFont(font_families[0]))
    app.setFont(QFont(font_families[0]))

    direct_launch_flags = {
        "--skip_settings",
        "--stream",
        "--camera",
        "--show_percent",
        "--show_score",
        "--fullscreen_mode",
        "--dark_theme",
        "--show_each_fps_label",
        "--dynamic_font_mode",
        "--min_font_size",
    }
    launch_direct = args.skip_settings or any(
        token.split("=")[0] in direct_launch_flags for token in sys.argv[1:]
    )

    if launch_direct:
        settings_window.apply_settings()
    else:
        settings_window.show()

    app_ret = app.exec_()

    sys.exit(app_ret)


if __name__ == "__main__":
    main()
