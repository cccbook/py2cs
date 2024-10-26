

https://www.facebook.com/allanyiin/posts/pfbid029xdWCGNKMwA9TP3dH8Hd3SQwTvFhwAJSq5CwwW95jCYVSmdLd1bmHH6t1MnGTFetl

```
關於我使用whisper轉錄課程錄音，其實只要3行代碼，前兩行安裝，轉錄只要一行
關鍵在於不要漏掉initial_prompt以及hallucination_silence_threshold，還有要用wav而非細節都喪失的mp3
!pip install git+https://github.com/kkroening/ffmpeg-python.git
!pip install git+https://github.com/openai/whisper.git
!whisper /content/drive/MyDrive/Colab/1020_2.WAV --language Chinese --model large-v3 --device cuda --initial_prompt "這是尹相志老師在大學教授\"生成式AI實務應用\"課程中關於「以文生圖」的授課內容，裡面會提到原理、如何利用ChatGPT, Copilot, Bing以及Capcut生成圖像以及生成圖像的prompt技巧" --hallucination_silence_threshold 2 --verbose True --threads 4
```

## log

```
(base) cccimac@cccimacdeiMac 06-whisper % pip install git+https://github.com/kkroening/ffmpeg-python.git
Collecting git+https://github.com/kkroening/ffmpeg-python.git
  Cloning https://github.com/kkroening/ffmpeg-python.git to /private/var/folders/c1/yg5q2n011t1315g8hjtfvmr40000gn/T/pip-req-build-jy1szivt
  Running command git clone --filter=blob:none --quiet https://github.com/kkroening/ffmpeg-python.git /private/var/folders/c1/yg5q2n011t1315g8hjtfvmr40000gn/T/pip-req-build-jy1szivt
  Resolved https://github.com/kkroening/ffmpeg-python.git to commit df129c7ba30aaa9ffffb81a48f53aa7253b0b4e6
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Collecting future (from ffmpeg-python==0.2.0)
  Downloading future-1.0.0-py3-none-any.whl.metadata (4.0 kB)
Downloading future-1.0.0-py3-none-any.whl (491 kB)
Building wheels for collected packages: ffmpeg-python
  Building wheel for ffmpeg-python (pyproject.toml) ... done
  Created wheel for ffmpeg-python: filename=ffmpeg_python-0.2.0-py3-none-any.whl size=25328 sha256=88918ef20630f411be6b6631f8209403ede6d184572023190d67d59616e3dd57
  Stored in directory: /private/var/folders/c1/yg5q2n011t1315g8hjtfvmr40000gn/T/pip-ephem-wheel-cache-3s_0fqvi/wheels/0f/ed/6d/22dc360efb116f1380ee6fa4d9e460f60518db616766ebaa1d
Successfully built ffmpeg-python
Installing collected packages: future, ffmpeg-python
Successfully installed ffmpeg-python-0.2.0 future-1.0.0
(base) cccimac@cccimacdeiMac 06-whisper % pip install git+https://github.com/openai/whisper.git
Collecting git+https://github.com/openai/whisper.git
  Cloning https://github.com/openai/whisper.git to /private/var/folders/c1/yg5q2n011t1315g8hjtfvmr40000gn/T/pip-req-build-wck7bbeo
  Running command git clone --filter=blob:none --quiet https://github.com/openai/whisper.git /private/var/folders/c1/yg5q2n011t1315g8hjtfvmr40000gn/T/pip-req-build-wck7bbeo
  Resolved https://github.com/openai/whisper.git to commit 25639fc17ddc013d56c594bfbf7644f2185fad84
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Collecting numba (from openai-whisper==20240930)
  Downloading numba-0.60.0-cp312-cp312-macosx_11_0_arm64.whl.metadata (2.7 kB)
Requirement already satisfied: numpy in /opt/miniconda3/lib/python3.12/site-packages (from openai-whisper==20240930) (1.26.4)
Requirement already satisfied: torch in /opt/miniconda3/lib/python3.12/site-packages (from openai-whisper==20240930) (2.4.1)
Requirement already satisfied: tqdm in /opt/miniconda3/lib/python3.12/site-packages (from openai-whisper==20240930) (4.66.4)
Collecting more-itertools (from openai-whisper==20240930)
  Downloading more_itertools-10.5.0-py3-none-any.whl.metadata (36 kB)
Collecting tiktoken (from openai-whisper==20240930)
  Downloading tiktoken-0.8.0-cp312-cp312-macosx_11_0_arm64.whl.metadata (6.6 kB)
Collecting llvmlite<0.44,>=0.43.0dev0 (from numba->openai-whisper==20240930)
  Downloading llvmlite-0.43.0-cp312-cp312-macosx_11_0_arm64.whl.metadata (4.8 kB)
Requirement already satisfied: regex>=2022.1.18 in /opt/miniconda3/lib/python3.12/site-packages (from tiktoken->openai-whisper==20240930) (2024.9.11)
Requirement already satisfied: requests>=2.26.0 in /opt/miniconda3/lib/python3.12/site-packages (from tiktoken->openai-whisper==20240930) (2.32.3)
Requirement already satisfied: filelock in /opt/miniconda3/lib/python3.12/site-packages (from torch->openai-whisper==20240930) (3.16.1)
Requirement already satisfied: typing-extensions>=4.8.0 in /opt/miniconda3/lib/python3.12/site-packages (from torch->openai-whisper==20240930) (4.12.2)
Requirement already satisfied: sympy in /opt/miniconda3/lib/python3.12/site-packages (from torch->openai-whisper==20240930) (1.13.3)
Requirement already satisfied: networkx in /opt/miniconda3/lib/python3.12/site-packages (from torch->openai-whisper==20240930) (3.3)
Requirement already satisfied: jinja2 in /opt/miniconda3/lib/python3.12/site-packages (from torch->openai-whisper==20240930) (3.1.4)
Requirement already satisfied: fsspec in /opt/miniconda3/lib/python3.12/site-packages (from torch->openai-whisper==20240930) (2024.6.1)
Requirement already satisfied: setuptools in /opt/miniconda3/lib/python3.12/site-packages (from torch->openai-whisper==20240930) (72.1.0)
Requirement already satisfied: charset-normalizer<4,>=2 in /opt/miniconda3/lib/python3.12/site-packages (from requests>=2.26.0->tiktoken->openai-whisper==20240930) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /opt/miniconda3/lib/python3.12/site-packages (from requests>=2.26.0->tiktoken->openai-whisper==20240930) (3.7)
Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/miniconda3/lib/python3.12/site-packages (from requests>=2.26.0->tiktoken->openai-whisper==20240930) (2.2.2)
Requirement already satisfied: certifi>=2017.4.17 in /opt/miniconda3/lib/python3.12/site-packages (from requests>=2.26.0->tiktoken->openai-whisper==20240930) (2024.8.30)
Requirement already satisfied: MarkupSafe>=2.0 in /opt/miniconda3/lib/python3.12/site-packages (from jinja2->torch->openai-whisper==20240930) (2.1.5)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/miniconda3/lib/python3.12/site-packages (from sympy->torch->openai-whisper==20240930) (1.3.0)
Downloading more_itertools-10.5.0-py3-none-any.whl (60 kB)
Downloading numba-0.60.0-cp312-cp312-macosx_11_0_arm64.whl (2.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.7/2.7 MB 1.8 MB/s eta 0:00:00
Downloading tiktoken-0.8.0-cp312-cp312-macosx_11_0_arm64.whl (982 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 982.6/982.6 kB 3.0 MB/s eta 0:00:00
Downloading llvmlite-0.43.0-cp312-cp312-macosx_11_0_arm64.whl (28.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 28.8/28.8 MB 5.8 MB/s eta 0:00:00
Building wheels for collected packages: openai-whisper
  Building wheel for openai-whisper (pyproject.toml) ... done
  Created wheel for openai-whisper: filename=openai_whisper-20240930-py3-none-any.whl size=803321 sha256=0e017d32045e9cc75e84dfc7eea2413e0a5039ec6309fc4f534149588d7138ff
  Stored in directory: /private/var/folders/c1/yg5q2n011t1315g8hjtfvmr40000gn/T/pip-ephem-wheel-cache-skc2zdgw/wheels/c3/03/25/5e0ba78bc27a3a089f137c9f1d92fdfce16d06996c071a016c
Successfully built openai-whisper
Installing collected packages: more-itertools, llvmlite, tiktoken, numba, openai-whisper
Successfully installed llvmlite-0.43.0 more-itertools-10.5.0 numba-0.60.0 openai-whisper-20240930 tiktoken-0.8.0
(base) cccimac@cccimacdeiMac 06-whisper % whisper
usage: whisper [-h] [--model MODEL] [--model_dir MODEL_DIR]
               [--device DEVICE] [--output_dir OUTPUT_DIR]
               [--output_format {txt,vtt,srt,tsv,json,all}]
               [--verbose VERBOSE] [--task {transcribe,translate}]
               [--language {af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,yue,zh,Afrikaans,Albanian,Amharic,Arabic,Armenian,Assamese,Azerbaijani,Bashkir,Basque,Belarusian,Bengali,Bosnian,Breton,Bulgarian,Burmese,Cantonese,Castilian,Catalan,Chinese,Croatian,Czech,Danish,Dutch,English,Estonian,Faroese,Finnish,Flemish,French,Galician,Georgian,German,Greek,Gujarati,Haitian,Haitian Creole,Hausa,Hawaiian,Hebrew,Hindi,Hungarian,Icelandic,Indonesian,Italian,Japanese,Javanese,Kannada,Kazakh,Khmer,Korean,Lao,Latin,Latvian,Letzeburgesch,Lingala,Lithuanian,Luxembourgish,Macedonian,Malagasy,Malay,Malayalam,Maltese,Mandarin,Maori,Marathi,Moldavian,Moldovan,Mongolian,Myanmar,Nepali,Norwegian,Nynorsk,Occitan,Panjabi,Pashto,Persian,Polish,Portuguese,Punjabi,Pushto,Romanian,Russian,Sanskrit,Serbian,Shona,Sindhi,Sinhala,Sinhalese,Slovak,Slovenian,Somali,Spanish,Sundanese,Swahili,Swedish,Tagalog,Tajik,Tamil,Tatar,Telugu,Thai,Tibetan,Turkish,Turkmen,Ukrainian,Urdu,Uzbek,Valencian,Vietnamese,Welsh,Yiddish,Yoruba}]
               [--temperature TEMPERATURE] [--best_of BEST_OF]
               [--beam_size BEAM_SIZE] [--patience PATIENCE]
               [--length_penalty LENGTH_PENALTY]
               [--suppress_tokens SUPPRESS_TOKENS]
               [--initial_prompt INITIAL_PROMPT]
               [--condition_on_previous_text CONDITION_ON_PREVIOUS_TEXT]
               [--fp16 FP16]
               [--temperature_increment_on_fallback TEMPERATURE_INCREMENT_ON_FALLBACK]
               [--compression_ratio_threshold COMPRESSION_RATIO_THRESHOLD]
               [--logprob_threshold LOGPROB_THRESHOLD]
               [--no_speech_threshold NO_SPEECH_THRESHOLD]
               [--word_timestamps WORD_TIMESTAMPS]
               [--prepend_punctuations PREPEND_PUNCTUATIONS]
               [--append_punctuations APPEND_PUNCTUATIONS]
               [--highlight_words HIGHLIGHT_WORDS]
               [--max_line_width MAX_LINE_WIDTH]
               [--max_line_count MAX_LINE_COUNT]
               [--max_words_per_line MAX_WORDS_PER_LINE] [--threads THREADS]
               [--clip_timestamps CLIP_TIMESTAMPS]
               [--hallucination_silence_threshold HALLUCINATION_SILENCE_THRESHOLD]
               audio [audio ...]
whisper: error: the following arguments are required: audio
(base) cccimac@cccimacdeiMac 06-whisper % whisper /Users/cccimac/Desktop/ccc/wav/cccTalk1.wav --language Chinese --model large-v3 --device cuda --initial_prompt "這是尹相志老師在大學教授\"生成式AI實務應用\"課程中關於「以文生圖」的授課內容，裡面會提到原理、如何利用ChatGPT, Copilot, Bing以及Capcut生成圖像以及生成圖像的prompt技巧" --hallucination_silence_threshold 2 --verbose True --threads 4 
100%|█████████████████████████████████████| 2.88G/2.88G [10:01<00:00, 5.13MiB/s]
/opt/miniconda3/lib/python3.12/site-packages/whisper/__init__.py:150: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(fp, map_location=device)
Traceback (most recent call last):
  File "/opt/miniconda3/bin/whisper", line 8, in <module>
    sys.exit(cli())
             ^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/whisper/transcribe.py", line 577, in cli
    model = load_model(model_name, device=device, download_root=model_dir)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/whisper/__init__.py", line 150, in load_model
    checkpoint = torch.load(fp, map_location=device)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/torch/serialization.py", line 1097, in load
    return _load(
           ^^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/torch/serialization.py", line 1525, in _load
    result = unpickler.load()
             ^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/torch/serialization.py", line 1492, in persistent_load
    typed_storage = load_tensor(dtype, nbytes, key, _maybe_decode_ascii(location))
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/torch/serialization.py", line 1466, in load_tensor
    wrap_storage=restore_location(storage, location),
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/torch/serialization.py", line 1389, in restore_location
    return default_restore_location(storage, map_location)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/torch/serialization.py", line 414, in default_restore_location
    result = fn(storage, location)
             ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/torch/serialization.py", line 391, in _deserialize
    device = _validate_device(location, backend_name)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/torch/serialization.py", line 364, in _validate_device
    raise RuntimeError(f'Attempting to deserialize object on a {backend_name.upper()} '
RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.
