# PDF-Vorleser

Liest lange PDFs mit natürlichen deutschen Neural-Stimmen vor, direkt auf dem Gerät.

- Installierbar als App: in Safari öffnen → Teilen → Zum Home-Bildschirm
- Läuft nach dem ersten Laden offline
- PDFs, Bibliothek und Lesestellen bleiben ausschließlich auf dem eigenen Gerät; es wird nichts hochgeladen
- Kapitelübersicht, Sleeptimer, Wiedergabe bei gesperrtem Bildschirm

## Stimmen

Die Stimmmodelle werden beim ersten Gebrauch direkt von Hugging Face geladen und nicht in diesem Repository verteilt:

- Hörbuch: [domoskanonos/piper-tts-models-de](https://huggingface.co/domoskanonos/piper-tts-models-de) (MIT)
- Anna, Clara, Lena, Marie, Sophie, Hanna: [Piper de_DE mls](https://huggingface.co/rhasspy/piper-voices) (CC BY 4.0, Multilingual LibriSpeech)
- Thorsten und Thorsten emotional: [Thorsten-Voice](https://huggingface.co/rhasspy/piper-voices) (CC0)
- Miro und Dii: [OpenVoiceOS](https://huggingface.co/OpenVoiceOS) (CC BY-NC-ND 4.0, nur nicht-kommerziell)

## Technik

[Piper](https://github.com/rhasspy/piper) über [piper-wasm](https://github.com/diffusion-studio/piper-wasm) (MIT) und [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) (MIT); PDF-Auslesen mit [PDF.js](https://github.com/mozilla/pdf.js) (Apache 2.0).
