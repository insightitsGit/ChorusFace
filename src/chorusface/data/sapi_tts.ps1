# Requires Windows PowerShell and System.Speech (built into Windows).
# Reads spoken text from stdin and writes a seekable WAVE payload to stdout.
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech

$text = [Console]::In.ReadToEnd()
if ([string]::IsNullOrWhiteSpace($text)) {
    [Console]::Error.WriteLine("sapi_tts: empty input")
    exit 2
}

$stream = New-Object System.IO.MemoryStream
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $rate = [Environment]::GetEnvironmentVariable("CHORUSFACE_SAPI_RATE")
    if (-not [string]::IsNullOrWhiteSpace($rate)) {
        $synth.Rate = [int]$rate
    }
    $voice = [Environment]::GetEnvironmentVariable("CHORUSFACE_SAPI_VOICE")
    if (-not [string]::IsNullOrWhiteSpace($voice)) {
        $null = $synth.SelectVoice($voice)
    }
    # MemoryStream is seekable, so Speak can patch the RIFF size headers.
    # Writing straight to stdout would leave a broken WAVE container.
    $synth.SetOutputToWaveStream($stream)
    $synth.Speak($text)
}
finally {
    $synth.Dispose()
}

$bytes = $stream.ToArray()
$stdout = [Console]::OpenStandardOutput()
$stdout.Write($bytes, 0, $bytes.Length)
$stdout.Flush()
