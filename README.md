# rwkv.zig
Minimal RWKV implementation in Zig

## How to run
1. Start and get inside the Docker container:
    ```bash
    cd infra-dev/
    docker-compose up -d
    docker-compose exec -it rwkv bash
    ```
2. Build tokenizers wrapper:
    ```bash
    cd ../tokenizers-wrapper/
    cargo build --release && mv target/release/libtokenizers.so /usr/local/lib/
    cbindgen --lang c --output tokenizers.h && mv tokenizers.h /usr/local/include/
    ```
3. Download models and export to another format:
    ```bash
    cd ../models/
    curl -L https://huggingface.co/BlinkDL/rwkv-4-pile-430m/resolve/main/RWKV-4-Pile-430M-20220808-8066.pth -o rwkv.pth
    curl -L https://github.com/BlinkDL/ChatRWKV/raw/main/20B_tokenizer.json -o tokenizer.json
    python3 export.py ./rwkv.pth ./
    ```
4. Inference:
    ```bash
    cd ../rwkv/
    zig build-exe ./src/main.zig -O ReleaseFast -lc -ltokenizers
    ./main ../models/rwkv.bin ../models/rwkv.json ../models/tokenizer.json
    ```
    Output:
    ```
    Hello darkness, my old friend. I am glad to see you. I have been looking for you for a long time. I have
    ```

## References
- [Minimal implementation of a relatively small (430m parameter) RWKV model which generates text](https://johanwind.github.io/2023/03/23/rwkv_details.html)
