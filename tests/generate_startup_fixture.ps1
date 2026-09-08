# Synthetic English speech for the real WASAPI startup regression.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$fixtureSpeech = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $englishVoice = $fixtureSpeech.GetInstalledVoices() | Where-Object {
        $_.Enabled -and $_.VoiceInfo.Culture.TwoLetterISOLanguageName -eq 'en'
    } | Select-Object -First 1
    if (-not $englishVoice) { throw 'La prueba necesita una voz de Windows en inglés.' }
    $fixtureSpeech.SelectVoice($englishVoice.VoiceInfo.Name)
    $fixtureRoot = Split-Path -Parent $PSScriptRoot
    $fixtureSpeech.SetOutputToWaveFile((Join-Path $fixtureRoot 'models\startup_regression.wav'))
    $fixtureSpeech.Speak('Press X to jump. Press the A button to continue. The press reported the news.')
    Write-Output "Voz de prueba: $($englishVoice.VoiceInfo.Name)"
} finally {
    $fixtureSpeech.Dispose()
}
