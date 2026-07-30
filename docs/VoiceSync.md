# Voice Sync

Every realtime language model already has a voice. What none of them ship is a
face that moves with the audio they are streaming out. This document describes
the channel that closes that gap, and — more importantly — the number that says
how well it closes it.

## What each side knows

The transcript says *what* is being said. The audio says *when*. Nothing in this
system recognises phonemes from a waveform.

That split is deliberate. Acoustic phoneme recognition is a whole research
problem, it needs a model, and it would be strictly worse than reading the text
the caller already has. So the viseme sequence comes from
`aiface.speech.tokenize_speech` — the same phonetic source of truth the offline
path uses — and the arriving energy decides only where each viseme lands.

The one consequence worth stating plainly: audio the transcript does not cover
closes the mouth rather than moving it. A channel that has ever been given words
will not fall back to guessing, because a guess is visible and a closed mouth is
merely early.

## The channel

**Preferred when the host already timed phonemes to its audio** (see
[PhoneticFidelity.md](PhoneticFidelity.md)):

```
POST /voice/timeline  {"caption":"...", "emotion":"happy",
                       "spans":[{"phoneme":"OU","start":0.0,"end":0.12}, ...]}
```

**When the host only has transcript + PCM:**

```
POST /voice/expect   {"text": "...", "emotion": "happy", "sample_rate": 24000}
POST /voice/pcm?format=pcm16&rate=24000   <raw bytes>
POST /voice/end
```

`expect` may be called repeatedly: realtime transcripts arrive as deltas, and
each delta extends the queue. `pcm` takes 16-bit little-endian mono PCM (what the
realtime APIs send) or `float32`; anything else is refused rather than guessed
at, because a wrong guess about layout desynchronises the whole face. A sample
split across two chunks is rejoined, since dropping one byte would shift every
sample after it.

These routes answer on the request thread. Every other bridge route queues a job
for the render thread, and for GPU work that is correct — but making a 20 ms
chunk wait for a frame would tie the audio clock to the frame rate. Alignment is
signal processing over numpy arrays and touches no GPU state, so the render loop
receives nothing but a list of timed viseme events.

## How the online aligner works

The offline aligner warps a script onto a clip by cumulative energy, which it can
do because it knows the clip's total energy in advance. Streaming has no such
luxury, so the same idea runs incrementally:

1. **Frames.** Arriving audio is reduced to one RMS value per 10 ms hop, over a
   25 ms window. Each frame is measured once and queued, so the state machine can
   look forward into audio that has already arrived without paying for it twice.
2. **Level and gate.** The speaker's level is tracked with a 0.85 s half-life over
   voiced frames only — letting silence pull it down would make the next word read
   as shouting and race through the script. A frame counts as voice at 28% of that
   level, floored by an absolute noise threshold.
3. **Spending.** Inside a phrase every frame spends against the articulation
   budget of the viseme currently held, blending energy with the clock. The next
   viseme is emitted the moment the budget runs out, so loud frames advance the
   script quickly and quiet frames crawl.
4. **Stops.** Where the script says the voice stops, no amount of energy may
   advance it. Only silence spends a full stop. That is what pins a long utterance
   to the pauses the speaker actually takes.
5. **Boundaries.** A gap in the voice and a relaxation in the script are two
   independent statements that the same boundary was reached. When they agree, the
   stretch that just ended measures how fast this voice really speaks, and the
   next stretch starts from a corrected estimate — so error cannot accumulate past
   a boundary.
6. **Hesitations.** A silence with too much script still owed is read as a pause
   inside a word, and nothing moves. If that quiet goes on far too long to be a
   hesitation, the words still queued were never going to get audio, and they are
   flushed at the moment the voice stopped rather than seconds later.

Two details exist because measurement said so, not because they read well.
Reading straight through punctuation disqualifies that phrase from measuring the
speaking rate: the audio just analysed covers more speech than the script it
would be divided by, and believing it pushes the estimate the wrong way — which
then paces the next phrase, and the next. And the energy blend is far lower than
the offline warp's, because the offline pass normalises against a clip it already
holds while the online one has only a running estimate.

## The oracle

Run one utterance through both paths and subtract. That is the entire idea, and
it is the only honest way to state streaming quality:

```bash
aiface-sync                                     # fixture set, local voice
aiface-sync --text "Yes." --wav clip.wav --detail
aiface-sync --json --budget-ms 250              # exit 1 if the budget is missed
```

| Number | What it means | Can a caller fix it? |
| --- | --- | --- |
| `bias` | Mean signed offset | Yes — one playback trim cancels it |
| `jitter` | Spread around that bias | No. This is the real quality measure |
| `trimmed p95` | 95th percentile error after the bias is removed | No — this is what a budget gates |
| `decision lag` | How late the channel committed, relative to the audio | Only by buffering that much audio |
| `coverage` | Fraction of offline visemes the streaming path also produced | — |

Matching the two sequences is where a careless oracle would flatter itself. The
paths normally agree exactly, but the offline pass drops spans that measured no
width and the streaming pass inserts a rest where the voice paused and the script
did not. A longest-matching-blocks diff over the viseme names keeps only pairs it
is sure about, so one inserted rest cannot shift every comparison after it.

The oracle is clock-free: it replays a clip in arrival order without sleeping. A
measurement therefore runs in milliseconds and is reproducible, which is what
makes it a CI gate instead of a demo.

## Measured

Windows SAPI, five fixture utterances, 20 ms chunks, 50 ms lookahead:

| Utterance | After trim (p95) | Jitter | Lag (p95) | Coverage |
| --- | --- | --- | --- | --- |
| `Hello there. I am listening to you now, carefully.` | 154.9 ms | 74.7 ms | 150 ms | 100% |
| `The lips follow the voice, not a guess about the voice.` | 64.5 ms | 30.9 ms | 105 ms | 100% |
| `Ask me anything, and watch the mouth land on every syllable.` | 194.7 ms | 124.2 ms | 108 ms | 100% |
| `Yes.` | 65.7 ms | 51.4 ms | 153 ms | 100% |
| 24-word sentence, one comma | 115.9 ms | 80.2 ms | 90 ms | 100% |

**Mean 119 ms, worst 195 ms after trim, 100% coverage**, against the 250 ms
budget `tests/test_sync.py` enforces whenever a local voice is installed.

Where the remaining error is: the opening phrase of an utterance pays for rate
calibration, since the channel has not yet heard this voice speak. After the
first boundary the middle of an utterance typically tracks within ±20 ms. The
worst line is the one whose transcript implies more syllables than the voice
articulates, which no amount of energy tracking can fix — only a boundary can.

## Tuning

Defaults were chosen by grid search over real synthesised speech, scored on mean
*and* worst-case trimmed p95 (a channel is only as good as the sentence it
handles least well), and checked on lines that were not in the scoring set. The
optimum is a broad plateau rather than a knife edge. Re-run the search after
changing how the channel spends energy:

```bash
python scripts/tune_voice_sync.py
```

| Knob | Default | Effect |
| --- | --- | --- |
| `--voice-lookahead` | 50 ms | Audio held back before a frame is judged. Steadier and later. |
| `--voice-trim` | 0 | Shifts streamed visemes against the caller's playback delay. |
| `--voice-rate` | 24000 | Sample rate assumed when a caller declares none. |
| `energy_blend` | 0.28 | How much of the advance follows energy rather than the clock. |
| `level_halflife` | 0.85 s | How fast the level estimate forgets. |
| `rate_trust` | 0.6 | Weight given to one freshly measured phrase. |
| `min_silence` | 80 ms | Silence that counts as a break rather than a stop consonant. |
| `long_silence` | 300 ms | Quiet that ends a phrase whatever the script believes. |

## The voice fixture

`--tts` is off by default. It exists because the oracle needs a clip with
knowable ground-truth alignment, and because a demo sometimes has no client
attached. Deleting it would delete the batch path, and with it the only way to
produce the number above.
