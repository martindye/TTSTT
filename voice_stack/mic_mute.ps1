<#
.SYNOPSIS
  Mute / unmute the default Windows microphone (input device).
.DESCRIPTION
  Works directly on the Windows audio endpoint (no extra dependencies).
  Safe to run any number of times; always reports the resulting state.
  Exit code 0 on success, non-zero on failure.
.EXAMPLE
  powershell -NoProfile -File C:\Users\press\OneDrive\Projects\TTSTT\voice_stack\mic_mute.ps1 mute
.EXAMPLE
  powershell -NoProfile -File C:\Users\press\OneDrive\Projects\TTSTT\voice_stack\mic_mute.ps1 unmute
.EXAMPLE
  powershell -NoProfile -File C:\Users\press\OneDrive\Projects\TTSTT\voice_stack\mic_mute.ps1 toggle
.EXAMPLE
  powershell -NoProfile -File C:\Users\press\OneDrive\Projects\TTSTT\voice_stack\mic_mute.ps1 status
#>
param(
    [ValidateSet('mute', 'unmute', 'toggle', 'status')]
    [string]$Action = 'status'
)

$ErrorActionPreference = 'Stop'

$code = @'
using System;
using System.Runtime.InteropServices;

namespace MicMute
{
    [Guid("D666888E-FC52-46A6-8D46-193C312597F9"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IMMDevice
    {
        [PreserveSig] int Activate([In] ref Guid rclsid, [In] IntPtr pActivateParams, [In] ref Guid riid, [Out] out IntPtr ppv);
        [PreserveSig] int OpenPropertyStore([In] uint stgmMode, [Out] out IntPtr ppStore);
        [PreserveSig] int GetId([Out, MarshalAs(UnmanagedType.LPWStr)] out string lpszId);
        [PreserveSig] int GetState([In] uint dwStateMask, [Out] out uint pdwState);
    }

    [Guid("A957A410-55C4-42DF-9A24-F29DBE128A78"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IMMDeviceEnumerator
    {
        [PreserveSig] int GetDefaultAudioEndpoint([In] uint eDataFlow, [In] uint eRole, [Out] out IMMDevice ppDevice);
    }

    [Guid("5CDF2C82-841C-4546-9B95-D446338CD66D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IAudioEndpointVolume
    {
        [PreserveSig] int GetChannelCount([Out] out uint pcChannels);
        [PreserveSig] int SetChannelVolume([In] uint nChannel, [In] float fLevel, [In] ref Guid pguidEventContext);
        [PreserveSig] int GetChannelVolume([In] uint nChannel, [Out] out float pfLevel);
        [PreserveSig] int SetMasterVolumeLevel([In] float fLevel, [In] ref Guid pguidEventContext);
        [PreserveSig] int SetMasterVolumeLevelScalar([In] float fLevel, [In] ref Guid pguidEventContext);
        [PreserveSig] int GetMasterVolumeLevel([Out] out float pfLevel);
        [PreserveSig] int GetMasterVolumeLevelScalar([Out] out float pfLevel);
        [PreserveSig] int SetMute([In, MarshalAs(UnmanagedType.Bool)] bool bMute, [In] ref Guid pguidEventContext);
        [PreserveSig] int GetMute([Out] out uint puMute);
    }

    public static class Mic
    {
        [DllImport("ole32.dll")]
        static extern int CoCreateInstance([In] ref Guid rclsid, IntPtr pvOuter, [In] uint dwClsContext, [In] ref Guid riid, [Out] out IntPtr ppv);

        [DllImport("ole32.dll")]
        static extern int CoInitializeEx(IntPtr pvReserved, uint dwCoInit);

        static IAudioEndpointVolume GetVolume()
        {
            // Activate via the .NET COM factory (raw CoCreateInstance fails in
            // pwsh 7's apartment; Type/Activator activation works reliably).
            Guid clsid = new Guid("BCDE0395-E52F-467C-8E3D-C4579291692E");
            IMMDeviceEnumerator en = (IMMDeviceEnumerator)Activator.CreateInstance(Type.GetTypeFromCLSID(clsid));
            IMMDevice dev;
            int hr2 = en.GetDefaultAudioEndpoint(1, 1, out dev); // eCapture, eConsole
            if (hr2 != 0) throw new Exception("no default capture endpoint 0x" + hr2.ToString("X8"));
            Guid iidVol = new Guid("5CDF2C82-841C-4546-9B95-D446338CD66D");
            IntPtr pVol;
            int hr3 = dev.Activate(ref iidVol, IntPtr.Zero, ref iidVol, out pVol);
            if (hr3 != 0) throw new Exception("activate volume endpoint failed 0x" + hr3.ToString("X8"));
            return (IAudioEndpointVolume)Marshal.GetObjectForIUnknown(pVol);
        }

        public static string State()
        {
            IAudioEndpointVolume vol = GetVolume();
            float lvl; vol.GetMasterVolumeLevelScalar(out lvl);
            uint mute; vol.GetMute(out mute);
            string s = mute == 1 ? "muted" : "unmuted";
            return s + " (level " + Math.Round(lvl * 100) + "%)";
        }

        public static string Set(bool wantMute)
        {
            IAudioEndpointVolume vol = GetVolume();
            Guid empty = Guid.Empty;
            vol.SetMute(wantMute, ref empty);
            float lvl; vol.GetMasterVolumeLevelScalar(out lvl);
            return (wantMute ? "mic muted" : "mic unmuted") + " (level " + Math.Round(lvl * 100) + "%)";
        }
    }
}
'@

Add-Type -TypeDefinition $code

switch ($Action) {
    'mute'   { [MicMute.Mic]::Set($true) }
    'unmute' { [MicMute.Mic]::Set($false) }
    'status' { [MicMute.Mic]::State() }
    'toggle' {
        $s = [MicMute.Mic]::State()
        if ($s -like 'muted*') { [MicMute.Mic]::Set($false) }
        else { [MicMute.Mic]::Set($true) }
    }
}
