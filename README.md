# PDF-Vorleser

Liest lange PDFs mit natürlichen deutschen Neural-Stimmen vor, direkt auf dem Gerät.

- Installierbar als App: in Safari öffnen → Teilen → Zum Home-Bildschirm
- Läuft nach dem ersten Laden offline
- PDFs, Bibliothek und Lesestellen bleiben ausschließlich auf dem eigenen Gerät; es wird nichts hochgeladen
- Kapitelübersicht, Sleeptimer, Wiedergabe bei gesperrtem Bildschirm

## Stimmen

- **Anna, Eve, Mary, Azelma, Jürgen, Michael, Giovanni, Javert:** [Pocket TTS](https://huggingface.co/kyutai/pocket-tts) von Kyutai, deutsches Modell (CC BY 4.0). Für den Browser nach ONNX umgewandelt (Sprachmodell 8-Bit, Decoder in 16-Bit gespeichert) und unter `pocket/` abgelegt, inklusive der vorberechneten Stimmprofile.
- **Hörbuch:** [domoskanonos/piper-tts-models-de](https://huggingface.co/domoskanonos/piper-tts-models-de) (MIT), wird beim ersten Gebrauch direkt von Hugging Face geladen.

## Technik

Pocket TTS läuft mit [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) (MIT), auf zwei Rechenkerne verteilt. Die Hörbuch-Stimme nutzt [Piper](https://github.com/rhasspy/piper) über [piper-wasm](https://github.com/diffusion-studio/piper-wasm) (MIT); PDF-Auslesen mit [PDF.js](https://github.com/mozilla/pdf.js) (Apache 2.0). Vorbereitete Bücher werden mit [lamejs](https://github.com/zhuker/lamejs) (LGPL-3.0, unverändert als `lame.min.js`) als MP3 auf dem Gerät gespeichert.
