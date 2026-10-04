# Coding voice recognition delay

The running bridge used fast Kyutai STT and default Pocket output. Qwen was
the language model in DSH; this failure was in the microphone bridge.

Before repair, logs showed repeated 120-second utterance timeouts sending
unfinished transcripts, including at 20:15:13 and 20:17:29. Endpoint code
counted SentencePiece word markers without first converting them to spaces,
and excluded unpunctuated utterances longer than twelve words from normal
pause completion. Both defects were present in origin/master at 968c02f.

The fix normalises word markers before counting, allows longer utterances
to finish after the existing six-second word gap, and waits for queued audio
before sending. Buffered frames now carry their own mailbox flag instead of
shared counters. A lock protects mailbox transfer and the queue accommodates
the complete thirty-second mailbox plus its existing live-audio capacity.
Timing and queue-drop diagnostics were added.

Validation: five endpoint/mailbox regression tests passed, and the bridge
compiled. With the user's Qwen server still running, the real GPU recogniser
processed 15.15 seconds of reference audio in 9.88 seconds; median frame time
was 52.96 ms and p95 was 60.49 ms against an 80 ms real-time budget. Raw results
are in stt_live_gpu_diagnostic.json. This checks decoder capacity under that
load, not the live microphone or conversational end-to-end latency.

Only the voice bridge was restarted. It reported ready at 20:21:55 and
continued heartbeats without errors. DSH and the user's Qwen server remained
running. Live conversational verification still requires the user speaking.

Existing limitations remain: microphone audio during speaker playback is
discarded to prevent feedback; busy-period recording keeps only the latest
thirty seconds for transcription. Pause detection uses recognised-word gaps,
not acoustic silence, and one- or two-word unpunctuated fragments still use
the longer timeout.

## Follow-up, 22:26–22:40

The first repair did not establish live multi-turn reliability. A later
Qwen session added a ninety-second continuous-RMS-triggered process restart,
but pauses reset that trigger and it never fired during the reported missed
attempts. At 22:27 the bridge was decoding frames without backlog or words.
That alone does not prove a corrupt model: room noise is not necessarily
speech.

During this investigation Windows explicitly reported the default Blue
Snowflake input muted (Core Audio GetMute returned 1). Two independent
capture APIs measured RMS about 0.000015 and peak 0.000031 while muted.
Unmuting it restored input RMS to 0.0233 and peak to 0.1545. The mute was
cleared once for this repair; startup does not override deliberate mute.
We cannot establish when the mute began or attribute every earlier missed
utterance to it. The requested spoken check received no user confirmation.

The bridge now recreates decoder streaming caches after replies and after
fifteen processed seconds without words. Wordless recovery retains and
retries up to fifteen seconds of unheard audio. It runs on the STT worker,
does not restart the process or reload weights, and does not require the
user to make ninety seconds of uninterrupted sound. Recognised words are
excluded from replay and endpointing waits for replay completion. Invalid
samples are sanitised before reaching the streaming model. Decoder errors
trigger a reset and frame retry. Logs now report raw word counts and RMS.

Validation: eleven offline regression tests pass, including three turn/end
cycles, stale stream chunks, retry after a short question and silence, no
duplicate recognised words, and no premature send during retry. The real
GPU model recognised the reference passage after three successive context
resets. A deliberately injected NaN poisoned its streaming state; automatic
reset and retained-audio retry recovered the full passage once, in 11.87
seconds of processing. This fault injection validates recovery, not the
original fault's cause. Allocated GPU memory stayed at 2463 MB across all
four cases. Results: stt_recovery_result.json; opt-in reproduction:
probe_stt_recovery.py. Four separate Pocket reply cycles and ninety seconds
of silence also passed the reference-audio check.

The bridge was restarted and ready at 22:39:51. DSH and the Qwen server
remained running. Live conversational verification remains outstanding.
The mailbox limit is currently sixty seconds (changed by the intervening
Qwen session), rather than the thirty seconds recorded in the first repair.
