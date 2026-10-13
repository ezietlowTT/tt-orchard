# TT-Metalium / TT-NN Models

## LLMs

| Model                                                         | Batch | Hardware                                                 | ttft (ms) | t/s/u | Target<br>t/s/u | t/s    | TT-Metalium Release 
|---------------------------------------------------------------|-------|----------------------------------------------------------|-----------|-------|-----------------|--------|---------------------
| [Qwen 3 32B (TP=8)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)                       | 32    | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 94
| [QwQ 32B (TP=8)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)                       | 32    | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 133  
| [DeepSeek R1 Distill Llama 3.3 70B (TP=8)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)       | 32    | [QuietBox  (Wormhole)](https://tenstorrent.com/hardware/tt-quietb
| [Llama 3.3 70B (TP=32)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/llama3_70b_galaxy)        | 32    | [Galaxy (Wormhole)](https://tenstorrent.com/hardware/galaxy) | 53      | 72
| [Llama 3.3 70B (TP=8)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)                 | 32    | [QuietBox  (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 159 
| [Llama 3.1 70B (TP=4)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)                 | 32    | [QuietBox (Blackhole)](https://tenstorrent.com/hardware/tt-quietbox) | 188*
| [Llama 3.2 11B Vision (TP=2)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)          | 16     | [n300 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 714 
| [Qwen 2.5 7B (TP=2)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)   | 32     | [n300 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 109      | 22.1  | 3
| [Qwen 2.5 72B (TP=8)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)               | 32    | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 223     
| [Falcon 7B](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/wormhole/falcon7b)                 | 32    | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 70      
| [Falcon 7B (DP=8)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/t3000/falcon7b)             | 256   | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 87      
| [Falcon 7B (DP=32)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/tg/falcon7b)               | 1024  | [Galaxy (Wormhole)](https://tenstorrent.com/hardware/galaxy)        | 121     
| [Falcon 40B (TP=8)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/t3000/falcon40b)           | 32    | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) |         
| [Llama 3.1 8B](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)                         | 32     | [p100 (Blackhole)](https://tenstorrent.com/hardware/blackhole)        | 61
| [Llama 3.1 8B](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)                         | 32     | [p150 (Blackhole)](https://tenstorrent.com/hardware/blackhole)        | 57
| [Llama 3.1 8B (DP=2)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)                  | 64     | [2 x p150 (Blackhole)](https://tenstorrent.com/hardware/blackhole)        
| [Llama 3.1 8B](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)                         | 32     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 104 
| [Llama 3.2 1B](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)                         | 32     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 23  
| [Llama 3.2 3B](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)                         | 32     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 52  
| [Mamba 2.8B](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/wormhole/mamba)                   | 32    | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 35      
| [Mistral 7B](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)                        | 32    | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        |  99 | 28
| [Mixtral 8x7B (TP=8)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)       | 32    | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 122       | 24.9

> **Last Update:** Dec 8, 2025
