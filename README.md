# Swiss German Fraud Call Scanner

A desktop app for detecting possible fraud or compliance-risk language in Swiss German call recordings and transcripts.

The app can:

- Load a keyword JSON file.
- Transcribe audio recordings into timestamped text.
- Analyse the transcript for fraud indicators.
- Highlight exact keyword matches, synonym/fuzzy matches, and indirect context matches.
- Display results in tabs for transcript, audio timeline, findings, and graph.
- Export scan results as JSON.

## Main Files

| File or folder | Purpose |
| --- | --- |
| `app.py` | Main GUI application. |
| `hackathon.py` | Fraud keyword, synonym, fuzzy, and context matching logic. |
| `audio_transcriber.py` | Audio-to-transcript logic using faster-whisper. |
| `Stichwortliste.json` | Main keyword JSON file. |
| `Synonymliste.example.json` | Example synonym JSON format. |
| `Audio/` | Folder for audio recordings. |
| `Transkript/` | Folder for generated or loaded transcripts. |
| `dist/FraudCallScanner/` | Built desktop app folder. |
| `dist/FraudCallScanner.zip` | Shareable ZIP package of the app. |

## How To Run The Packaged App

1. Open:

   ```text
   dist/FraudCallScanner/FraudCallScanner.exe
   ```

2. Or share this ZIP file:

   ```text
   dist/FraudCallScanner.zip
   ```

3. If sharing the app, send the whole ZIP. The app needs the included `_internal` folder because it contains the Python runtime, GUI libraries, faster-whisper files, and the bundled tiny Whisper model.

## How To Use The App

1. Select the keyword JSON file.

   Usually this is:

   ```text
   Stichwortliste.json
   ```

2. Select audio input.

   You can choose:

   - one audio file
   - multiple audio files
   - a whole audio folder

3. Choose speech settings.

   - `tiny`: fastest, lower accuracy
   - `base`: better accuracy, slower
   - `small`: more accurate, slower
   - `medium`: strongest option in the app, slowest

4. Click `Transcribe + Analyse`.

   The app will:

   - convert audio into timestamped transcript text
   - display the transcript in Tab 1
   - run fraud detection on that transcript
   - show findings in Tab 2
   - show a bar graph in Tab 3

5. To analyse edited transcript text, change the text in the transcript box and click `Analyse Transcript`.

## Result Colors

| Match type | Color | Meaning |
| --- | --- | --- |
| Exact | Red | A keyword from the JSON was directly found. |
| Synonym | Blue | A synonym, fuzzy spelling, or close wording was found. |
| Context | Yellow | Nearby lines together suggest the keyword meaning indirectly. |

## Audio Timeline

The audio timeline shows where suspicious parts occur in the recording.

After transcription and analysis:

- click `Audio Timeline`
- press play to listen
- colored segments show where matches happen by timestamp

## Minimum Confidence

Minimum confidence controls how strict the scanner is.

- Lower value, for example `0.30`: shows more possible matches, but can include more false positives.
- Higher value, for example `0.80`: shows fewer matches, but they are more likely to be relevant.

Exact keyword matches normally have high confidence. Synonym, fuzzy, and context matches depend more on the selected threshold.

## Accent-Insensitive Mode

Accent-insensitive mode makes matching more forgiving for German and Swiss German spellings.

For example, it helps compare words written with accents or expanded spellings such as:

```text
ae, oe, ue, e
```

This is useful when transcripts are generated from speech, because ASR output may not always preserve accents consistently.

## Refresh

Click `Refresh` to clear:

- transcript text
- audio timeline
- fraud findings
- graph
- result counters

It does not delete your files.

## Export Results

Click `Export Results JSON` to save the detected findings to a JSON file.

The export is useful for:

- submitting hackathon output
- checking detected lines later
- comparing different confidence settings

## Building The App Again

If you edit the source code and want to rebuild the Windows app, run:

```powershell
powershell -ExecutionPolicy Bypass -File build_app.ps1
```

The build output will be created in:

```text
dist/FraudCallScanner/
```

The shareable ZIP is:

```text
dist/FraudCallScanner.zip
```

## Important Notes

- The app is a detection aid, not a final legal or compliance decision system.
- Context and synonym matches should be reviewed by a person.
- Audio transcription accuracy depends on recording quality, background noise, speaker accent, and selected model size.
- The bundled app includes the tiny Whisper model so it can transcribe without separately sharing model files.

