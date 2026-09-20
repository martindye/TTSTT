"""One-off: verify GatewaySession.inject reaches the GUI gateway."""
import sys

sys.path.insert(0, ".")
from pathlib import Path

from voice_stack.handoff import (find_target_session, load_browser_secret,
                                sessions_dir_for)
from voice_stack.coding_voice import GatewaySession

gui_home = Path(r"C:\Users\press\.dsh")
secret = load_browser_secret(gui_home)
assert secret, "no secret"
sdir = sessions_dir_for(gui_home)
assert sdir, "no sessions dir"
sid = find_target_session(sdir)
print("target session:", sid)
gw = GatewaySession("http://127.0.0.1:3080", sid, "127.0.0.1:3080", secret)
gw.inject("Coding-voice bridge self-test — please ignore this message. (not typed by the user)")
print("inject ok")
