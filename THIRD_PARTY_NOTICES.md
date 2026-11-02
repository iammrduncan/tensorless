# Third-party notices

tensorless is licensed under Apache-2.0 (see `LICENSE` and `NOTICE`). The engine in `engine/` contains code from the projects below. Each keeps its own license. The license texts are in `LICENSES/`, `engine/LICENSE`, `engine/licenses/` and the vendored directories named here. Engine images carry the same files under `/opt/tensorless/`.

## The engine's lineage

| Project | License | Where its notice is |
|---|---|---|
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp), the ggml authors | MIT | `engine/LICENSE`, `LICENSES/llama.cpp-MIT.txt` |
| [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant), TheTom | MIT (the same notice) | `engine/LICENSE` |
| [llamAmpere](https://github.com/JakeATX/llamAmpere), JakeATX | MIT (the same notice) | `engine/LICENSE` |
| [exllamav3](https://github.com/turboderp-org/exllamav3), Turboderp (portions, for the EXL3 weight types) | MIT | `engine/licenses/LICENSE-exllamav3`, `LICENSES/exllamav3-MIT.txt` |

## Vendored into the engine (`engine/vendor/`)

| Library | License | Notice |
|---|---|---|
| [cpp-httplib](https://github.com/yhirose/cpp-httplib), yhirose | MIT | `engine/vendor/cpp-httplib/LICENSE` |
| [nlohmann/json](https://github.com/nlohmann/json), Niels Lohmann | MIT | `engine/licenses/LICENSE-jsonhpp`, `LICENSES/nlohmann-json-MIT.txt` |
| [miniaudio](https://github.com/mackron/miniaudio), David Reid | Public domain or MIT-0 | the end of `engine/vendor/miniaudio/miniaudio.h` |
| [stb_image](https://github.com/nothings/stb), Sean Barrett | Public domain or MIT | the end of `engine/vendor/stb/stb_image.h` |
| [subprocess.h](https://github.com/sheredom/subprocess.h), sheredom | Unlicense (public domain) | the head of `engine/vendor/sheredom/subprocess.h` |
| [xxHash](https://github.com/Cyan4973/xxHash), Yann Collet | BSD-2-Clause | `engine/vendor/hash/xxhash/LICENSE` |
| SHA-256, Igor Pavlov | Public domain | `engine/vendor/hash/sha256/LICENSE` |
| SHA-1, Steve Reid | Public domain | the head of `engine/vendor/hash/sha1/sha1.h` |
| [rotate-bits](https://github.com/jb55/rotate-bits.h), William Casarin | MIT | `engine/vendor/hash/rotate-bits/LICENSE.md` |

## Runtime

Engine images are built on NVIDIA's CUDA base images (`nvidia/cuda`, CUDA 12.6). Their CUDA libraries are covered by the NVIDIA CUDA Toolkit license, which comes with the image.

The launcher depends on [BurntSushi/toml](https://github.com/BurntSushi/toml) (MIT).

Model weights are not part of tensorless. Each model's license is named in its recipe.
