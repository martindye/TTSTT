"""Mute / unmute / report the default Windows microphone.

Pure ctypes (no dependencies). Talks to the Windows audio stack through the
standard COM vtables of MMDevApi.dll, so no interface GUIDs beyond the two
invariant ones (IUnknown, IAudioEndpointVolume) are needed.

Usage:  mic_mute.py status | mute | unmute
Prints one line, e.g. "unmuted (level 100%)" or "muted (level 100%)".
Exit code 0 on success, 1 on failure.
"""
import sys
from ctypes import (CFUNCTYPE, POINTER, Structure, byref, c_char, c_float,
                    c_int32, c_uint16, c_uint32, c_void_p, windll)


class GUID(Structure):
    _fields_ = [("Data1", c_uint32), ("Data2", c_uint16), ("Data3", c_uint16),
                ("Data4", c_char * 8)]


def g(s):
    h = s.replace("-", "")
    return GUID(int(h[0:8], 16), int(h[8:12], 16), int(h[12:16], 16),
                bytes.fromhex(h[16:32]))


IID_IUNKNOWN = g("00000000-0001-0000-C000-000000000046")
CLSID_ENUMERATOR = g("BCDE0395-E52F-467C-8E3D-C4579291692E")
IID_VOL = g("5CDF2C82-841C-4546-9B95-D446338CD66D")  # IAudioEndpointVolume

ole32 = windll.ole32
ole32.CoCreateInstance.argtypes = [POINTER(GUID), c_void_p, c_uint32,
                                   POINTER(GUID), POINTER(c_void_p)]
ole32.CoCreateInstance.restype = c_int32

QI = CFUNCTYPE(c_int32, c_void_p, POINTER(GUID), POINTER(c_void_p))
GET_DEFAULT = CFUNCTYPE(c_int32, c_void_p, c_uint32, c_uint32, POINTER(c_void_p))
ACTIVATE = CFUNCTYPE(c_int32, c_void_p, POINTER(GUID), c_void_p,
                     POINTER(GUID), POINTER(c_void_p))
GET_MUTE = CFUNCTYPE(c_int32, c_void_p, POINTER(c_uint32))
SET_MUTE = CFUNCTYPE(c_int32, c_void_p, c_uint32, c_void_p)
GET_LEVEL = CFUNCTYPE(c_int32, c_void_p, POINTER(c_float))


def die(msg):
    print(f"mic: {msg}", file=sys.stderr)
    sys.exit(1)


def slot(obj, n):
    return cast(obj, POINTER(c_void_p))[n]


def main(action):
    pp = c_void_p()
    hr = ole32.CoCreateInstance(byref(CLSID_ENUMERATOR), None, 0x17,
                                byref(IID_IUNKNOWN), byref(pp))
    if hr != 0 or not pp.value:
        die(f"CoCreateInstance failed: 0x{hr & 0xFFFFFFFF:08X}")
    enumerator = pp.value

    # IMMDeviceEnumerator::GetDefaultAudioEndpoint(eCapture=1, eConsole=1)
    devp = c_void_p()
    hr = GET_DEFAULT(slot(enumerator, 3))(enumerator, 1, 1, byref(devp))
    if hr != 0 or not devp.value:
        die(f"GetDefaultAudioEndpoint failed: 0x{hr & 0xFFFFFFFF:08X}")
    device = devp.value

    # IMMDevice::Activate(IAudioEndpointVolume)
    volp = c_void_p()
    hr = ACTIVATE(slot(device, 3))(device, byref(IID_VOL), None,
                                   byref(IID_VOL), byref(volp))
    if hr != 0 or not volp.value:
        die(f"Activate(IAudioEndpointVolume) failed: 0x{hr & 0xFFFFFFFF:08X}")
    vol = volp.value

    muted = c_uint32(0)
    GET_MUTE(slot(vol, 11))(vol, byref(muted))
    was_muted = bool(muted.value & 1)
    level = c_float(0.0)
    GET_LEVEL(slot(vol, 9))(vol, byref(level))

    if action == "mute" and not was_muted:
        SET_MUTE(slot(vol, 10))(vol, 1, None)
    elif action == "unmute" and was_muted:
        SET_MUTE(slot(vol, 10))(vol, 0, None)

    now_muted = (action == "mute") or (was_muted and action != "unmute")
    state = "muted" if now_muted else "unmuted"
    print(f"{state} (level {level.value * 100:.0f}%)")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if action not in ("status", "mute", "unmute"):
        die(f"unknown action {action!r} (use status|mute|unmute)")
    if action == "toggle":
        die("use status then mute or unmute")
    main(action)
